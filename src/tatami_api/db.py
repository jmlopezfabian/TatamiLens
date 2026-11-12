import json
import uuid
from datetime import UTC, datetime

from sqlalchemy import event
from sqlalchemy.engine import Engine
from sqlmodel import Field, SQLModel, create_engine


def new_id() -> str:
    return uuid.uuid4().hex[:12]


def now() -> datetime:
    return datetime.now(UTC)


class VideoRow(SQLModel, table=True):
    __tablename__ = "video"
    id: str = Field(default_factory=new_id, primary_key=True)
    filename: str
    stored_name: str
    duration_s: float | None = None
    fps: float | None = None
    created_at: datetime = Field(default_factory=now)


class AnalysisRow(SQLModel, table=True):
    __tablename__ = "analysis"
    id: str = Field(default_factory=new_id, primary_key=True)
    video_id: str = Field(index=True, foreign_key="video.id")
    status: str = "queued"
    progress: float = 0.0
    backend: str
    model: str
    sample_fps: float = 2.0
    error: str | None = None
    created_at: datetime = Field(default_factory=now)
    result_json: str | None = None

    @property
    def result(self) -> dict | None:
        return json.loads(self.result_json) if self.result_json else None


def make_engine(url: str) -> Engine:
    engine = create_engine(url, connect_args={"check_same_thread": False})

    @event.listens_for(engine, "connect")
    def _pragmas(conn, _):  # noqa: ANN001
        cur = conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL")
        cur.execute("PRAGMA foreign_keys=ON")
        cur.close()

    SQLModel.metadata.create_all(engine)
    return engine
