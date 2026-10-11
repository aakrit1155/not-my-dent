import io
import time

from fastapi.testclient import TestClient
from PIL import Image

from app.main import app


def _poll(client, job_id):
    for _ in range(60):
        payload = client.get(f"/api/jobs/{job_id}").json()
        if payload["state"] == "complete":
            return payload["report"]
        if payload["state"] == "failed":
            raise AssertionError(payload)
        time.sleep(.1)
    raise AssertionError("job timed out")


def test_demo_end_to_end():
    with TestClient(app) as client:
        assert client.get("/").status_code == 200
        assert client.get("/healthz").json()["mode"] == "mock"
        created = client.post("/api/verify-claim", data={"demo": "true", "claim_text": "Front bumper dent"})
        assert created.status_code == 202
        report = _poll(client, created.json()["job_id"])
        assert report["mode"] == "mock"
        assert report["claim_valid"] is False
        proof = client.get(report["proof_image_url"])
        assert proof.status_code == 200 and proof.content.startswith(b"\x89PNG\r\n\x1a\n")
        assert proof.headers["cache-control"] == "no-store"


def test_uploaded_photo_and_videos_end_to_end():
    buffer = io.BytesIO()
    Image.new("RGB", (180, 120), "orange").save(buffer, "PNG")
    files = {
        "video_before": ("before.mp4", b"dummy video bytes", "video/mp4"),
        "video_after": ("after.mp4", b"dummy video bytes", "video/mp4"),
        "claim_image": ("claim.png", buffer.getvalue(), "image/png"),
    }
    with TestClient(app) as client:
        created = client.post("/api/verify-claim", data={"claim_text": "Bumper mark"}, files=files)
        assert created.status_code == 202
        report = _poll(client, created.json()["job_id"])
        assert report["claim_text"] == "Bumper mark"
        assert client.get(report["proof_image_url"]).status_code == 200


def test_validation():
    with TestClient(app) as client:
        response = client.post("/api/verify-claim", data={"claim_text": "A dent"})
        assert response.status_code == 422
        response = client.post("/api/verify-claim", data={"demo": "true", "claim_text": " "})
        assert response.status_code == 422
