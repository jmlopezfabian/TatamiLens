import os

import uvicorn

from .app import create_app

app = create_app()


def run() -> None:
    uvicorn.run("tatami_api.main:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))  # noqa: S104
