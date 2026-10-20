"""Main annotation orchestration.

For every image/frame from `dataset_parser_service`, runs `inference.infer()`,
converts the detections to Ultralytics YOLO label format, and writes the
image/label pair to a staging area. Knows nothing about LocateAnything's
implementation -- only consumes `inference.infer()`'s output.
"""
from __future__ import annotations

import logging
from pathlib import Path

from core.base_config import config
from core.service.dataset_parser_service import iter_source_images
from inference import Detection, infer

logger = logging.getLogger(__name__)

STAGING_DIRNAME = "_staging"


def staging_dir() -> Path:
    """Where freshly generated image/label pairs land before splitting."""
    return Path(config.output_dataset_path) / STAGING_DIRNAME


def generate_dataset() -> None:
    """Annotate every image/frame under INPUT_PATH and stage YOLO labels.

    Raises:
        ValueError: If PROMPTS is empty (should already be caught by
            `config.validate()`, checked again here as a safety net).
    """
    if not config.prompts:
        raise ValueError("PROMPTS must list at least one class to annotate.")

    images_dir = staging_dir() / "images"
    labels_dir = staging_dir() / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    class_to_index = {name: idx for idx, name in enumerate(config.prompts)}

    frame_count = 0
    skipped_count = 0
    for name, image in iter_source_images(config.input_path, config.frame_skip, config.max_frames):
        try:
            detections = infer(image, config.prompts)
            detections = [d for d in detections if d.confidence >= config.confidence_threshold]
            image.save(images_dir / f"{name}.jpg")
            _write_label_file(labels_dir / f"{name}.txt", detections, class_to_index, image.width, image.height)
        except Exception:
            logger.exception("Skipping frame %r after annotation failure.", name)
            skipped_count += 1
            continue

        frame_count += 1
        if frame_count % 10 == 0:
            logger.info("Annotated %d frames", frame_count)

    logger.info("Finished annotating %d frames (%d skipped) into %s", frame_count, skipped_count, staging_dir())


def _write_label_file(
    path: Path,
    detections: list[Detection],
    class_to_index: dict[str, int],
    width: int,
    height: int,
) -> None:
    lines = []
    for detection in detections:
        class_idx = class_to_index[detection.class_name]
        x1, y1, x2, y2 = detection.bbox
        cx = ((x1 + x2) / 2) / width
        cy = ((y1 + y2) / 2) / height
        w = (x2 - x1) / width
        h = (y2 - y1) / height
        lines.append(f"{class_idx} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    path.write_text("\n".join(lines), encoding="utf-8")
