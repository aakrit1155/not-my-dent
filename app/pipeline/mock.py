"""Synthetic preview only. No input video is decoded or inspected."""

import io
import random
import threading
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch
from PIL import Image, ImageOps, UnidentifiedImageError

_RENDER_LOCK = threading.Lock()  # Matplotlib's pyplot state is not thread safe.


def _vehicle_panel(ax, *, label: str, after: bool, rng: random.Random):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.set_facecolor("#e9f1f4")
    ax.add_patch(plt.Rectangle((0, 0), 10, 1.35, color="#d2dbe0"))
    ax.add_patch(FancyBboxPatch((1.1, 1.8), 7.8, 2.1, boxstyle="round,pad=0.15,rounding_size=0.7", facecolor="#547d8b", edgecolor="#254956", linewidth=3))
    ax.add_patch(plt.Polygon([[3.0, 3.8], [4.1, 5.2], [7.4, 5.2], [8.3, 3.8]], color="#547d8b"))
    ax.add_patch(plt.Polygon([[4.1, 3.9], [4.65, 4.9], [6.0, 4.9], [6.0, 3.9]], color="#c5e3ec"))
    ax.add_patch(plt.Polygon([[6.2, 3.9], [6.2, 4.9], [7.28, 4.9], [7.95, 3.9]], color="#c5e3ec"))
    for x in (3.0, 7.25):
        ax.add_patch(Circle((x, 1.92), 0.62, color="#1e3541"))
        ax.add_patch(Circle((x, 1.92), 0.28, color="#d2e1e7"))
    ax.add_patch(Circle((1.65, 2.92), 0.14, color="#f4c983"))
    ax.add_patch(Circle((8.52, 2.9), 0.12, color="#efb29b"))
    # Identical highlighted defect in both mock frames, slightly shifted framing.
    x = 8.46 + (0.035 if after else 0)
    ax.plot([x - .32, x, x + .19], [2.46, 2.27, 2.35], color="#f8d0bd", lw=3)
    ax.add_patch(Circle((x, 2.36), 0.62, fill=False, edgecolor="#f26954", linewidth=3))
    ax.text(.36, 6.45, label, color="#193b49", fontsize=13, weight="bold")
    ax.text(.36, 5.9, f"SIMULATED FRAME  #{rng.randint(340, 440):04d}", color="#466778", fontsize=9)
    ax.axis("off")


def process_claim_pipeline(
    video_before: Path | None,
    video_after: Path | None,
    claim_text: str,
    claim_image: Path | None,
) -> tuple[dict, bytes]:
    """Integration seam: return (JSON-serializable report, PNG bytes).

    The paths exist only while this function runs. They are deleted immediately
    afterward. Replace this function with real inference; keep its return shape.
    """
    del video_before, video_after  # The sample processor intentionally ignores videos.
    time.sleep(2.0)  # Make progress visible without loading any ML models.
    rng = random.Random()
    with _RENDER_LOCK:
        return _render_report(claim_text, claim_image, rng)


def _render_report(claim_text: str, claim_image: Path | None, rng: random.Random) -> tuple[dict, bytes]:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4), dpi=150)
    fig.patch.set_facecolor("#f6f9fa")
    try:
        _vehicle_panel(axes[0], label="01  Before pickup", after=False, rng=rng)
        _vehicle_panel(axes[1], label="02  After drop-off", after=True, rng=rng)
        axes[2].set_facecolor("#e9f1f4")
        if claim_image is not None:
            try:
                with Image.open(claim_image) as source:
                    image = ImageOps.exif_transpose(source)
                    image.thumbnail((1600, 1200))
                    axes[2].imshow(image.convert("RGB"))
            except (UnidentifiedImageError, OSError, ValueError) as exc:
                raise ValueError("The vendor photo is not a supported image.") from exc
            axes[2].add_patch(Circle((.78, .68), .14, transform=axes[2].transAxes, fill=False, edgecolor="#f26954", linewidth=3))
        else:
            _vehicle_panel(axes[2], label="03  Vendor photo (sample)", after=False, rng=rng)
        if claim_image is not None:
            axes[2].text(.03, .94, "03  Vendor photo", transform=axes[2].transAxes, fontsize=13, weight="bold", color="white", bbox={"facecolor": "#173b48", "pad": 8, "edgecolor": "none"})
            axes[2].axis("off")
        fig.suptitle("Illustrative evidence comparison  |  MOCK OUTPUT", color="#163543", fontsize=18, weight="bold", y=.99)
        fig.text(.5, .045, "Synthetic before/after frames • Uploaded videos were not analyzed • Highlighting is illustrative", ha="center", fontsize=10, color="#496575")
        fig.subplots_adjust(left=.025, right=.975, bottom=.10, top=.90, wspace=.035)
        output = io.BytesIO()
        fig.savefig(output, format="png", dpi=150, facecolor=fig.get_facecolor())
        png = output.getvalue()
    finally:
        plt.close(fig)
    return {
        "status": "success",
        "mode": "mock",
        "claim_valid": False,
        "verdict": "Pre-existing damage detected (demo)",
        "explanation": "In this simulated example, the same bumper mark appears before pickup and after drop-off. The claim would be disputed if actual frame matching confirmed this observation. No video analysis was performed.",
        "confidence": 0.942,
        "confidence_label": "Simulated confidence",
        "claim_text": claim_text,
        "disclaimer": "Illustrative mock result. Uploaded videos were not inspected; this is not a real claim decision.",
    }, png
