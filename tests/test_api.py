from conftest import wait_done

MOCK = {"backend": "mock", "model": "mock", "sample_fps": 2.0}


def test_health(client):
    assert client.get("/api/health").json() == {"status": "ok"}


def test_backends(client):
    names = {b["name"] for b in client.get("/api/backends").json()}
    assert "mock" in names


def test_upload_and_list(client, video):
    assert video["filename"] == "clip.mp4"
    assert video["fps"] == 10.0
    assert abs(video["duration_s"] - 10.0) < 0.2
    assert [v["id"] for v in client.get("/api/videos").json()] == [video["id"]]
    assert client.get(f"/api/videos/{video['id']}").json()["id"] == video["id"]
    assert client.get("/api/videos/nope").status_code == 404


def test_stream_range(client, video, mp4):
    url = f"/api/videos/{video['id']}/stream"
    data = mp4.read_bytes()
    full = client.get(url)
    assert full.status_code == 200 and full.content == data
    assert full.headers["accept-ranges"] == "bytes"
    r = client.get(url, headers={"Range": "bytes=10-99"})
    assert r.status_code == 206
    assert r.content == data[10:100]
    assert r.headers["content-range"] == f"bytes 10-99/{len(data)}"
    r = client.get(url, headers={"Range": "bytes=-20"})
    assert r.status_code == 206 and r.content == data[-20:]
    r = client.get(url, headers={"Range": f"bytes={len(data) - 5}-"})
    assert r.content == data[-5:]
    assert client.get(url, headers={"Range": "bytes=999999999-"}).status_code == 416


def test_analysis_mock(client, video):
    r = client.post(f"/api/videos/{video['id']}/analyses", json=MOCK)
    assert r.status_code == 201
    a = r.json()
    assert a["status"] == "queued" and a["result"] is None
    done = wait_done(client, a["id"])
    assert done["status"] == "done" and done["progress"] == 1.0
    res = done["result"]
    assert res["schema_version"] == "1"
    assert res["frames"] and res["segments"] and set(res["score"]) == {"A", "B"}
    lst = client.get(f"/api/videos/{video['id']}/analyses").json()
    assert [x["id"] for x in lst] == [a["id"]]


def test_analysis_errors(client, video):
    assert client.post("/api/videos/nope/analyses", json=MOCK).status_code == 404
    bad = client.post(f"/api/videos/{video['id']}/analyses", json={**MOCK, "backend": "zzz"})
    assert bad.status_code == 422
    assert client.get("/api/analyses/nope").status_code == 404


def test_labels_and_rescore(client, video):
    vid = video["id"]
    assert client.get(f"/api/videos/{vid}/labels").status_code == 404
    a = client.post(f"/api/videos/{vid}/analyses", json=MOCK).json()
    wait_done(client, a["id"])
    assert client.post(f"/api/analyses/{a['id']}/rescore").status_code == 404

    labels = {
        "video": "clip.mp4",
        "athletes": {"A": "gi blanco", "B": "gi azul"},
        "intervals": [
            {"start": 0, "end": 4, "position": "standing", "top": "none", "source": "human"},
            {"start": 4, "end": 10, "position": "side_control", "top": "B", "source": "human"},
        ],
    }
    assert client.put(f"/api/videos/{vid}/labels", json=labels).status_code == 200
    assert client.get(f"/api/videos/{vid}/labels").json() == labels
    exp = client.get(f"/api/videos/{vid}/labels/export")
    assert exp.status_code == 200 and exp.json() == labels
    assert "clip.labels.json" in exp.headers["content-disposition"]

    r = client.post(f"/api/analyses/{a['id']}/rescore")
    assert r.status_code == 200
    res = r.json()
    assert res["score"]["B"] > 0 and res["score"]["A"] == 0
    assert any(e["kind"] == "takedown" and e["athlete"] == "B" for e in res["events"])
    assert client.get(f"/api/analyses/{a['id']}").json()["result"]["score"] == res["score"]


def test_labels_validation(client, video):
    bad = {
        "video": "x",
        "intervals": [
            {"start": 0, "end": 3, "position": "standing", "top": "A", "source": "human"}
        ],
    }
    assert client.put(f"/api/videos/{video['id']}/labels", json=bad).status_code == 422
    bad["intervals"][0].update(top="none", end=0)
    assert client.put(f"/api/videos/{video['id']}/labels", json=bad).status_code == 422


def test_delete(client, video, tmp_path):
    vid = video["id"]
    a = client.post(f"/api/videos/{vid}/analyses", json=MOCK).json()
    wait_done(client, a["id"])
    client.put(f"/api/videos/{vid}/labels", json={"video": "clip.mp4", "intervals": []})
    assert client.delete(f"/api/videos/{vid}").status_code == 204
    assert client.get(f"/api/videos/{vid}").status_code == 404
    assert client.get(f"/api/analyses/{a['id']}").status_code == 404
    assert client.get("/api/videos").json() == []
    assert not list((tmp_path / "data" / "videos").iterdir())
    assert not list((tmp_path / "data" / "labels").iterdir())
    assert client.delete(f"/api/videos/{vid}").status_code == 404


def test_cors(client):
    r = client.get("/api/health", headers={"Origin": "http://localhost:5173"})
    assert r.headers["access-control-allow-origin"] == "http://localhost:5173"


def test_uses_real_tatami_vision_when_installed():
    import importlib.util

    from tatami_api import vision

    assert vision.USING_REAL_VISION == (importlib.util.find_spec("tatami_vision") is not None)


def test_backends_propagate_extra_fields(client):
    from tatami_api import vision

    items = {b["name"]: b for b in client.get("/api/backends").json()}
    assert items["mock"]["available"] is True
    if vision.USING_REAL_VISION:
        assert set(items) == {"ollama", "anthropic", "local", "mock"}
        assert "default_model" in items["mock"] and "detail" in items["mock"]
        assert items["ollama"]["default_model"] == "qwen3-vl:8b"


def test_rescore_with_real_or_fallback(client, video):
    a = client.post(f"/api/videos/{video['id']}/analyses", json=MOCK).json()
    wait_done(client, a["id"])
    labels = {"video": "clip.mp4", "athletes": {"A": "x", "B": "y"},
              "intervals": [{"start": 0, "end": 10, "position": "mount", "top": "B",
                             "source": "human"}]}  # fmt: skip
    assert client.put(f"/api/videos/{video['id']}/labels", json=labels).status_code == 200
    r = client.post(f"/api/analyses/{a['id']}/rescore")
    assert r.status_code == 200
    assert {s["position"] for s in r.json()["segments"]} == {"mount"}
    assert r.json()["score"]["A"] == 0


def test_fallback_without_tatami_vision(monkeypatch, tmp_path):
    from tatami_api import vision

    monkeypatch.setattr(vision, "_tv", None)
    assert any(b["name"] == "mock" for b in vision.list_backends())
    res = vision.analyze_video(tmp_path / "nofile.mp4", backend="mock", model="mock")
    assert res["schema_version"] == "1" and res["frames"]
