import time

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from tatami_api.app import create_app
from tatami_api.config import Settings


def make_mp4(path, seconds=2, fps=10):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (64, 48))
    for i in range(seconds * fps):
        w.write(np.full((48, 64, 3), i % 255, dtype=np.uint8))
    w.release()
    return path


@pytest.fixture
def client(tmp_path):
    app = create_app(Settings(data_dir=tmp_path / "data"))
    with TestClient(app) as c:
        yield c


@pytest.fixture
def mp4(tmp_path):
    return make_mp4(tmp_path / "clip.mp4", seconds=10)


@pytest.fixture
def video(client, mp4):
    with open(mp4, "rb") as f:
        r = client.post("/api/videos", files={"file": ("clip.mp4", f, "video/mp4")})
    assert r.status_code == 201
    return r.json()


def wait_done(client, aid, timeout=10):
    end = time.time() + timeout
    while time.time() < end:
        a = client.get(f"/api/analyses/{aid}").json()
        if a["status"] in ("done", "error"):
            return a
        time.sleep(0.05)
    raise TimeoutError
