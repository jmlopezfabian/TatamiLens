import json
import mimetypes
import os
import shutil
import uuid
from collections.abc import Iterator
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
from fastapi import APIRouter, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
from sqlmodel import Session, select

from . import vision
from .config import Settings
from .db import AnalysisRow, VideoRow, make_engine
from .schemas import (
    Analysis,
    AnalysisCreate,
    AnalysisResult,
    BackendInfo,
    Health,
    Labels,
    Video,
)
from .worker import AnalysisWorker

CHUNK = 1024 * 1024


def probe_video(path: Path) -> tuple[float | None, float | None]:
    """(duration_s, fps) con cv2; None si no se puede leer."""
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            return None, None
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        n = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        if fps <= 0:
            return None, None
        return (round(n / fps, 3) if n > 0 else None), round(float(fps), 3)
    finally:
        cap.release()


def analysis_out(a: AnalysisRow, with_result: bool = True) -> Analysis:
    return Analysis(
        id=a.id, video_id=a.video_id, status=a.status, progress=a.progress,
        backend=a.backend, model=a.model, error=a.error, created_at=a.created_at,
        result=a.result if with_result else None,
    )  # fmt: skip


def _parse_range(header: str, size: int) -> tuple[int, int] | None:
    try:
        unit, _, spec = header.partition("=")
        if unit.strip() != "bytes" or "," in spec:
            return None
        start_s, _, end_s = spec.strip().partition("-")
        if start_s == "":
            n = int(end_s)
            if n <= 0:
                return None
            return max(size - n, 0), size - 1
        start = int(start_s)
        end = int(end_s) if end_s else size - 1
        end = min(end, size - 1)
        if start > end or start >= size:
            return None
        return start, end
    except ValueError:
        return None


def _iter_file(path: Path, start: int, end: int) -> Iterator[bytes]:
    with open(path, "rb") as f:
        f.seek(start)
        remaining = end - start + 1
        while remaining > 0:
            data = f.read(min(CHUNK, remaining))
            if not data:
                break
            remaining -= len(data)
            yield data


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    settings.videos_dir.mkdir(parents=True, exist_ok=True)
    settings.labels_dir.mkdir(parents=True, exist_ok=True)
    engine = make_engine(settings.db_url)
    worker = AnalysisWorker(engine, settings.videos_dir, settings.max_concurrent_analyses)
    worker.recover()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        worker.shutdown()

    app = FastAPI(title="TatamiLens API", version="0.1.0", lifespan=lifespan)
    app.state.settings, app.state.engine, app.state.worker = settings, engine, worker
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["Content-Range", "Accept-Ranges", "Content-Length"],
    )
    api = APIRouter(prefix="/api")

    def get_video(s: Session, video_id: str) -> VideoRow:
        v = s.get(VideoRow, video_id)
        if v is None:
            raise HTTPException(404, "video no encontrado")
        return v

    def get_analysis(s: Session, analysis_id: str) -> AnalysisRow:
        a = s.get(AnalysisRow, analysis_id)
        if a is None:
            raise HTTPException(404, "análisis no encontrado")
        return a

    def labels_path(video_id: str) -> Path:
        return settings.labels_dir / f"{video_id}.json"

    @api.get("/health", response_model=Health)
    def health() -> Health:
        return Health()

    @api.get("/backends", response_model=list[BackendInfo])
    def backends() -> list[dict]:
        return vision.list_backends()

    @api.post("/videos", response_model=Video, status_code=201)
    def upload_video(file: UploadFile = File(...)) -> Video:  # noqa: B008
        original = Path(file.filename or "video").name or "video"
        stored = f"{uuid.uuid4().hex}{Path(original).suffix.lower()}"
        dest = settings.videos_dir / stored
        with open(dest, "wb") as out:
            shutil.copyfileobj(file.file, out, CHUNK)
        duration, fps = probe_video(dest)
        row = VideoRow(filename=original, stored_name=stored, duration_s=duration, fps=fps)
        with Session(engine) as s:
            s.add(row)
            s.commit()
            s.refresh(row)
            return Video(**row.model_dump())

    @api.get("/videos", response_model=list[Video])
    def list_videos() -> list[Video]:
        with Session(engine) as s:
            rows = s.exec(select(VideoRow).order_by(VideoRow.created_at.desc())).all()  # type: ignore[attr-defined]
            return [Video(**r.model_dump()) for r in rows]

    @api.get("/videos/{video_id}", response_model=Video)
    def read_video(video_id: str) -> Video:
        with Session(engine) as s:
            return Video(**get_video(s, video_id).model_dump())

    @api.get("/videos/{video_id}/stream")
    def stream_video(video_id: str, request: Request) -> Response:
        with Session(engine) as s:
            path = settings.videos_dir / get_video(s, video_id).stored_name
        if not path.exists():
            raise HTTPException(404, "archivo de video no encontrado")
        size = path.stat().st_size
        mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        base = {"Accept-Ranges": "bytes"}
        rng = request.headers.get("range")
        if rng is None:
            return StreamingResponse(_iter_file(path, 0, size - 1), media_type=mime,
                                     headers={**base, "Content-Length": str(size)})  # fmt: skip
        parsed = _parse_range(rng, size)
        if parsed is None:
            return Response(status_code=416, headers={**base, "Content-Range": f"bytes */{size}"})
        start, end = parsed
        return StreamingResponse(
            _iter_file(path, start, end), status_code=206, media_type=mime,
            headers={**base, "Content-Range": f"bytes {start}-{end}/{size}",
                     "Content-Length": str(end - start + 1)},
        )  # fmt: skip

    @api.delete("/videos/{video_id}", status_code=204)
    def delete_video(video_id: str) -> Response:
        with Session(engine) as s:
            v = get_video(s, video_id)
            for a in s.exec(select(AnalysisRow).where(AnalysisRow.video_id == video_id)):
                s.delete(a)
            stored = v.stored_name
            s.delete(v)
            s.commit()
        (settings.videos_dir / stored).unlink(missing_ok=True)
        labels_path(video_id).unlink(missing_ok=True)
        return Response(status_code=204)

    @api.post("/videos/{video_id}/analyses", response_model=Analysis, status_code=201)
    def create_analysis(video_id: str, body: AnalysisCreate) -> Analysis:
        names = {b["name"] for b in vision.list_backends()}
        if body.backend not in names:
            raise HTTPException(422, f"backend desconocido: {body.backend}. Disponibles: "
                                     f"{sorted(names)}")  # fmt: skip
        with Session(engine) as s:
            get_video(s, video_id)
            a = AnalysisRow(video_id=video_id, backend=body.backend, model=body.model,
                            sample_fps=body.sample_fps)  # fmt: skip
            s.add(a)
            s.commit()
            s.refresh(a)
            out = analysis_out(a)
        worker.submit(out.id)
        return out

    @api.get("/videos/{video_id}/analyses", response_model=list[Analysis])
    def list_analyses(video_id: str) -> list[Analysis]:
        with Session(engine) as s:
            get_video(s, video_id)
            rows = s.exec(select(AnalysisRow).where(AnalysisRow.video_id == video_id)
                          .order_by(AnalysisRow.created_at.desc())).all()  # type: ignore[attr-defined]  # fmt: skip
            return [analysis_out(a) for a in rows]

    @api.get("/analyses/{analysis_id}", response_model=Analysis)
    def read_analysis(analysis_id: str) -> Analysis:
        with Session(engine) as s:
            return analysis_out(get_analysis(s, analysis_id))

    @api.put("/videos/{video_id}/labels", response_model=Labels)
    def put_labels(video_id: str, labels: Labels) -> Labels:
        with Session(engine) as s:
            get_video(s, video_id)
        p = labels_path(video_id)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(labels.model_dump_json(indent=2), encoding="utf-8")
        os.replace(tmp, p)
        return labels

    @api.get("/videos/{video_id}/labels", response_model=Labels)
    def get_labels(video_id: str) -> Labels:
        with Session(engine) as s:
            get_video(s, video_id)
        p = labels_path(video_id)
        if not p.exists():
            raise HTTPException(404, "sin etiquetas corregidas")
        return Labels.model_validate_json(p.read_text(encoding="utf-8"))

    @api.get("/videos/{video_id}/labels/export")
    def export_labels(video_id: str) -> FileResponse:
        with Session(engine) as s:
            v = get_video(s, video_id)
            name = f"{Path(v.filename).stem}.labels.json"
        p = labels_path(video_id)
        if not p.exists():
            raise HTTPException(404, "sin etiquetas corregidas")
        return FileResponse(p, media_type="application/json", filename=name)

    @api.post("/analyses/{analysis_id}/rescore", response_model=AnalysisResult)
    def rescore_analysis(analysis_id: str) -> dict:
        with Session(engine) as s:
            a = get_analysis(s, analysis_id)
            if a.status != "done" or a.result is None:
                raise HTTPException(409, "el análisis aún no está terminado")
            p = labels_path(a.video_id)
            if not p.exists():
                raise HTTPException(404, "sin etiquetas corregidas para este video")
            labels = json.loads(p.read_text(encoding="utf-8"))
            new = vision.rescore(a.result, labels["intervals"])
            a.result_json = json.dumps(new)
            s.add(a)
            s.commit()
            return new

    app.include_router(api)
    return app
