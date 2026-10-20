"""Optional visual QA utility: draws YOLO label boxes back onto images.

Not required for the production pipeline -- only runs when
VISUALIZE_DATASET=true.
"""
from __future__ import annotations

from pathlib import Path

import cv2

from core.base_config import config


def visualize_dataset(dataset_dir: str, output_dir: str) -> None:
    """Render every image in dataset_dir with its YOLO label boxes drawn on.

    Args:
        dataset_dir: Root containing images/ and labels/ (flat or split).
        output_dir: Where annotated preview images are written.
    """
    images_dir = Path(dataset_dir) / "images"
    labels_dir = Path(dataset_dir) / "labels"
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for image_path in sorted(images_dir.rglob("*.*")):
        label_path = (labels_dir / image_path.relative_to(images_dir)).with_suffix(".txt")
        if not label_path.exists():
            continue

        image = cv2.imread(str(image_path))
        height, width = image.shape[:2]
        for line in label_path.read_text(encoding="utf-8").splitlines():
            class_idx_str, cx_str, cy_str, w_str, h_str = line.split()
            class_idx = int(class_idx_str)
            cx, cy, w, h = float(cx_str), float(cy_str), float(w_str), float(h_str)
            x1, y1 = int((cx - w / 2) * width), int((cy - h / 2) * height)
            x2, y2 = int((cx + w / 2) * width), int((cy + h / 2) * height)

            label = config.prompts[class_idx] if class_idx < len(config.prompts) else str(class_idx)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, label, (x1, max(y1 - 5, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        out_path = out_dir / image_path.relative_to(images_dir)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), image)
