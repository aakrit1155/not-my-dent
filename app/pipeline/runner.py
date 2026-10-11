"""Bridge between the HTTP layer and the NotMyDent engine.

Models are loaded once (preload at startup) and reused. One GPU job runs at a
time; other jobs wait on the lock. The engine is imported lazily so the web app
still starts on machines without the ML stack.
"""

import logging
import threading
from pathlib import Path

log = logging.getLogger("notmydent")
_GPU_LOCK = threading.Lock()
STATUS = {"models": "not loaded", "error": None}


def preload():
    """Load YOLO, CLIP and Qwen3-VL into GPU memory once."""
    with _GPU_LOCK:
        STATUS["models"] = "loading"
        try:
            from app.pipeline import engine

            engine.load_detector().to(engine.DEVICE)
            engine.load_clip()
            engine.load_vlm()
            STATUS["models"] = "ready"
        except Exception as exc:
            log.exception("Model preload failed")
            STATUS["models"] = "failed"
            STATUS["error"] = str(exc)


def run(claim_image: Path, video_before: Path, video_after: Path, claim_text: str, output_dir: Path) -> dict:
    with _GPU_LOCK:
        from app.pipeline import engine

        return engine.run_notmydent(
            claim_photo=str(claim_image),
            before_path=str(video_before),
            after_path=str(video_after),
            claim=claim_text,
            output_dir=str(output_dir),
            show=False,
        )
