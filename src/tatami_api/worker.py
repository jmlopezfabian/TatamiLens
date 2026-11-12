"""Cola de análisis en background (ThreadPoolExecutor, 1 worker por defecto)."""

import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor

from sqlmodel import Session, select

from . import vision
from .db import AnalysisRow, VideoRow

log = logging.getLogger(__name__)


class AnalysisWorker:
    def __init__(self, engine, videos_dir, max_workers: int = 1):
        self.engine = engine
        self.videos_dir = videos_dir
        self.pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="analysis")

    def recover(self) -> None:
        """Marca como error lo que quedó queued/running de una ejecución anterior."""
        with Session(self.engine) as s:
            for a in s.exec(
                select(AnalysisRow).where(AnalysisRow.status.in_(["queued", "running"]))
            ):
                a.status, a.error = "error", "interrumpido por reinicio del servidor"
                s.add(a)
            s.commit()

    def submit(self, analysis_id: str) -> None:
        self.pool.submit(self._run, analysis_id)

    def shutdown(self) -> None:
        self.pool.shutdown(wait=False, cancel_futures=True)

    def _update(self, analysis_id: str, **fields) -> None:
        with Session(self.engine) as s:
            a = s.get(AnalysisRow, analysis_id)
            if a is None:
                return
            for k, v in fields.items():
                setattr(a, k, v)
            s.add(a)
            s.commit()

    def _run(self, analysis_id: str) -> None:
        with Session(self.engine) as s:
            a = s.get(AnalysisRow, analysis_id)
            if a is None:  # borrado mientras estaba en cola
                return
            v = s.get(VideoRow, a.video_id)
            if v is None:
                return
            path = self.videos_dir / v.stored_name
            backend, model, fps = a.backend, a.model, a.sample_fps
        self._update(analysis_id, status="running", progress=0.0)
        lock = threading.Lock()
        last = [0.0]

        def cb(p: float) -> None:
            p = max(0.0, min(1.0, float(p)))
            with lock:
                if p - last[0] < 0.01 and p < 1.0:
                    return
                last[0] = p
            self._update(analysis_id, progress=p)

        try:
            result = vision.analyze_video(path, backend=backend, model=model, sample_fps=fps,
                                          progress_cb=cb)  # fmt: skip
            self._update(analysis_id, status="done", progress=1.0, result_json=json.dumps(result))
        except Exception as e:  # noqa: BLE001
            log.exception("análisis %s falló", analysis_id)
            self._update(analysis_id, status="error", error=f"{type(e).__name__}: {e}")
