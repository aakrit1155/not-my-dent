"""HTTP flow tests. The GPU engine is replaced with a fake that returns the same report shape."""

import io
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app
from app.pipeline import runner


def _png() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (180, 120), "orange").save(buffer, "PNG")
    return buffer.getvalue()


def fake_run(claim_image, video_before, video_after, claim_text, output_dir):
    assert Path(claim_image).is_file() and Path(video_before).is_file() and Path(video_after).is_file()
    folder = Path(output_dir) / "claim_01"
    folder.mkdir(parents=True)
    for name in ("comparison", "company_frame", "company_crop", "after_frame", "after_crop"):
        (folder / f"{name}.png").write_bytes(_png())
    side = lambda phase: {"frame": str(folder / f"{phase}_frame.png"), "crop": str(folder / f"{phase}_crop.png")}
    score = {"full_frame": {"type": "scratch", "confidence": 0.81}, "zoom": None, "zoom_source_px": [100, 80]}
    return {
        "claim": claim_text,
        "overall_verdict": "INCONCLUSIVE",
        "findings": [{
            "claim_id": "claim_01",
            "verdict": "INCONCLUSIVE",
            "company": {"description": "Scratches", "confidence": score, "images": side("company")},
            "before": {"timestamp_seconds": None, "status": "unclear", "description": "No reliable matching location.", "confidence": None, "images": None},
            "after": {"timestamp_seconds": 3.25, "status": "same", "description": "Scratches", "confidence": score, "images": side("after")},
            "comparison": "Insufficient comparable evidence.",
            "figure_path": str(folder / "comparison.png"),
            "comparisons": {},
        }],
    }


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(runner, "preload", lambda: None)
    monkeypatch.setattr(runner, "run", fake_run)
    with TestClient(app) as test_client:
        yield test_client


def _files():
    return {
        "video_before": ("before.mp4", b"dummy video bytes", "video/mp4"),
        "video_after": ("after.mp4", b"dummy video bytes", "video/mp4"),
        "claim_image": ("claim.png", _png(), "image/png"),
    }


def _poll(client, job_id):
    for _ in range(60):
        payload = client.get(f"/api/jobs/{job_id}").json()
        if payload["state"] == "complete":
            return payload["report"]
        if payload["state"] == "failed":
            raise AssertionError(payload)
        time.sleep(.1)
    raise AssertionError("job timed out")


def test_upload_end_to_end(client):
    assert client.get("/").status_code == 200
    assert client.get("/healthz").json()["status"] == "ok"
    created = client.post("/api/verify-claim", data={"claim_text": "Scratches on the front bumper"}, files=_files())
    assert created.status_code == 202
    job_id = created.json()["job_id"]
    report = _poll(client, job_id)
    assert report["claim"] == "Scratches on the front bumper"
    finding = report["findings"][0]
    assert finding["figure_path"] == f"/api/jobs/{job_id}/files/claim_01/comparison.png"
    assert finding["after"]["images"]["crop"] == f"/api/jobs/{job_id}/files/claim_01/after_crop.png"
    image = client.get(finding["company"]["images"]["frame"])
    assert image.status_code == 200 and image.content.startswith(b"\x89PNG\r\n\x1a\n")
    assert client.get(f"/api/jobs/{job_id}/files/../inputs/claim.image").status_code == 404


def test_engine_failure_is_reported(client, monkeypatch):
    def broken(*args):
        raise ValueError("No video frames extracted: /tmp/secret/path")
    monkeypatch.setattr(runner, "run", broken)
    created = client.post("/api/verify-claim", data={"claim_text": "Dent"}, files=_files())
    for _ in range(60):
        payload = client.get(f"/api/jobs/{created.json()['job_id']}").json()
        if payload["state"] == "failed":
            assert "/tmp" not in payload["error"]
            return
        time.sleep(.1)
    raise AssertionError("job did not fail")


def test_validation(client):
    assert client.post("/api/verify-claim", data={"claim_text": "A dent"}).status_code == 422
    assert client.post("/api/verify-claim", data={"claim_text": " "}, files=_files()).status_code == 422
