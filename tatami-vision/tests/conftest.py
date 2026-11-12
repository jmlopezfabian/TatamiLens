import cv2
import numpy as np
import pytest


@pytest.fixture
def synthetic_video(tmp_path):
    path = tmp_path / "synthetic.mp4"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (160, 120))
    for i in range(100):  # 10 s
        img = np.full((120, 160, 3), (i * 2) % 255, np.uint8)
        cv2.circle(img, (20 + i, 60), 15, (0, 255, 0), -1)
        w.write(img)
    w.release()
    return path
