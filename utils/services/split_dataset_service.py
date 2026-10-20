"""Splits the staged, freshly-annotated dataset (optionally merged with a
previous dataset) into Ultralytics YOLO train/val/test folders and writes
data.yaml.
"""
from __future__ import annotations

import logging
import random
import shutil
from pathlib import Path

import yaml

from core.base_config import config
from core.service.annotator_service import staging_dir
from core.service.dataset_parser_service import IMAGE_EXTENSIONS

logger = logging.getLogger(__name__)


def build_final_dataset() -> Path:
    """Merge (if configured) and split the dataset, writing data.yaml.

    Returns:
        Path to the written data.yaml.

    Raises:
        ValueError: If no annotated image/label pairs were found at all.
    """
    pairs = _collect_pairs(staging_dir())

    if not config.is_new_dataset:
        # Previous pairs go first so any filename collision renames the *new* file,
        # never silently overwriting or renaming away the pre-existing dataset.
        pairs = _collect_pairs(Path(config.previous_dataset_path)) + pairs

    if not pairs:
        raise ValueError("No annotated image/label pairs found to build a dataset from.")

    random.Random(config.split_seed).shuffle(pairs)

    n_train = int(len(pairs) * config.train_split)
    n_val = int(len(pairs) * config.val_split)
    splits = {
        "train": pairs[:n_train],
        "val": pairs[n_train:n_train + n_val],
        "test": pairs[n_train + n_val:],
    }

    final_dir = Path(config.output_dataset_path)
    used_names: set[str] = set()
    for split_name, split_pairs in splits.items():
        if not split_pairs:
            continue
        images_dir = final_dir / "images" / split_name
        labels_dir = final_dir / "labels" / split_name
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        for image_path, label_path in split_pairs:
            dest_name = _dedupe_name(image_path.stem, used_names)
            shutil.copy2(image_path, images_dir / f"{dest_name}{image_path.suffix}")
            shutil.copy2(label_path, labels_dir / f"{dest_name}.txt")

    data_yaml_path = _write_data_yaml(final_dir, splits)
    shutil.rmtree(staging_dir(), ignore_errors=True)
    logger.info(
        "Final dataset: %d train / %d val / %d test -> %s",
        len(splits["train"]), len(splits["val"]), len(splits["test"]), final_dir,
    )
    return data_yaml_path


def _dedupe_name(stem: str, used_names: set[str]) -> str:
    """Return a filename stem guaranteed unique within this build, handling collisions."""
    candidate = stem
    suffix = 1
    while candidate in used_names:
        suffix += 1
        candidate = f"{stem}_{suffix}"
    used_names.add(candidate)
    return candidate


def _collect_pairs(dataset_dir: Path) -> list[tuple[Path, Path]]:
    """Recursively collect (image, label) pairs under dataset_dir/images and /labels.

    Works for both a flat layout (images/*.jpg) and an already-split one
    (images/train/*.jpg), since it walks the tree instead of assuming depth.
    """
    images_dir = dataset_dir / "images"
    labels_dir = dataset_dir / "labels"
    if not images_dir.is_dir():
        return []

    pairs = []
    for image_path in sorted(images_dir.rglob("*.*")):
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        label_path = (labels_dir / image_path.relative_to(images_dir)).with_suffix(".txt")
        if label_path.exists():
            pairs.append((image_path, label_path))
        else:
            logger.warning("No label file for %s, skipping.", image_path)
    return pairs


def _write_data_yaml(final_dir: Path, splits: dict[str, list]) -> Path:
    data_yaml_path = final_dir / "data.yaml"
    data_yaml_path.write_text(
        yaml.safe_dump(
            {
                "path": str(final_dir),
                "train": "images/train",
                "val": "images/val" if splits["val"] else "images/train",
                "test": "images/test" if splits["test"] else "",
                "names": {idx: name for idx, name in enumerate(config.prompts)},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return data_yaml_path
