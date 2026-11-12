from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="TATAMI_", extra="ignore")

    data_dir: Path = Path("./data")
    cors_origins: str = "http://localhost:5173"
    max_concurrent_analyses: int = Field(default=1, ge=1)

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def videos_dir(self) -> Path:
        return self.data_dir / "videos"

    @property
    def labels_dir(self) -> Path:
        return self.data_dir / "labels"

    @property
    def db_url(self) -> str:
        return f"sqlite:///{(self.data_dir / 'tatami.db').resolve()}"
