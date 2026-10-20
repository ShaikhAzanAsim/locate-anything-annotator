"""Central configuration hub.

Every environment variable the pipeline reads is declared here, with a default
and (where required) validation. Every other module imports the `config`
singleton from this file instead of calling `os.getenv` directly.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"
if _ENV_FILE.exists():
    load_dotenv(_ENV_FILE)


def _get_bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


def _get_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _get_float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def _get_list(name: str, default: str = "") -> list[str]:
    return [item.strip() for item in os.getenv(name, default).split(",") if item.strip()]


@dataclass(frozen=True)
class Config:
    """Typed, validated view of every environment variable the pipeline uses."""

    # --- input (single image, single video, folder of images, or folder of videos) ---
    input_path: str = os.getenv("INPUT_PATH", "")
    frame_skip: int = _get_int("FRAME_SKIP", 30)
    max_frames: int = _get_int("MAX_FRAMES", 0)  # 0 = no cap (per video, if input is a folder of videos)

    # --- LocateAnything prompts (also doubles as the YOLO class order) ---
    prompts: list = field(default_factory=lambda: _get_list("OBJECT"))
    prompt_template: str = os.getenv(
        "PROMPT_TEMPLATE", "Locate all the instances that match the following description: {categories}."
    )
    # NOTE: LocateAnything-3B is a generative VLM that emits <box> coordinate tokens; it does not
    # produce a calibrated per-detection confidence score. Every detection is stubbed at
    # confidence=1.0, so this threshold is currently a no-op. Kept for interface compatibility in
    # case a future model/version exposes real scores.
    confidence_threshold: float = _get_float("CONFIDENCE_THRESHOLD", 0.0)

    # --- LocateAnything model ---
    model_name: str = os.getenv("MODEL_NAME", "nvidia/LocateAnything-3B")
    # Local snapshot dir: checked on startup and downloaded into if missing (see
    # utils/services/weight_dowload.py), so re-running the container never re-downloads.
    model_local_dir: str = os.getenv("MODEL_LOCAL_DIR", "./weight/LocateAnything-3B")
    device: str = os.getenv("DEVICE", "cuda")
    generation_mode: str = os.getenv("GENERATION_MODE", "hybrid")
    generation_temperature: float = _get_float("GENERATION_TEMPERATURE", 0.7)
    max_new_tokens: int = _get_int("MAX_NEW_TOKENS", 8192)

    # --- dataset output / merge ---
    output_dataset_path: str = os.getenv("OUTPUT_DATASET_PATH", "/app/dataset")
    is_new_dataset: bool = _get_bool("IS_NEW_DATASET", True)
    previous_dataset_path: str = os.getenv("PREVIOUS_DATASET_PATH", "")
    image_size: int = _get_int("IMAGE_SIZE", 640)
    train_split: float = _get_float("TRAIN_SPLIT", 0.8)
    val_split: float = _get_float("VAL_SPLIT", 0.1)
    test_split: float = _get_float("TEST_SPLIT", 0.1)
    split_seed: int = _get_int("SPLIT_SEED", 42)

    # --- optional visual QA pass (draws label boxes onto images) ---
    visualize_dataset: bool = _get_bool("VISUALIZE_DATASET", False)
    visualization_dir: str = os.getenv("VISUALIZATION_DIR", "/app/dataset_preview")

    # --- YOLO (ultralytics) training ---
    yolo_model: str = os.getenv("YOLO_MODEL", "yolo11n.pt")
    epochs: int = _get_int("EPOCHS", 100)
    batch: int = _get_int("BATCH", 16)
    workers: int = _get_int("WORKERS", 8)
    save_period: int = _get_int("SAVE_PERIOD", -1)
    project_name: str = os.getenv("PROJECT_NAME", "/app/dataset/runs")
    run_name: str = os.getenv("RUN_NAME", "train")
    weights_dir: str = os.getenv("WEIGHTS_DIR", "/app/weight")
    # Escape hatch for any ultralytics train() kwarg not covered by a named var above
    # (e.g. {"lr0": 0.001, "patience": 20}), merged in with lower precedence than the named ones.
    extra_train_args: dict = field(default_factory=lambda: json.loads(os.getenv("EXTRA_TRAIN_ARGS", "{}")))

    def validate(self) -> None:
        """Fail fast on missing/invalid required configuration.

        Raises:
            ValueError: If a required variable is missing or a value is out of range.
        """
        if not self.input_path:
            raise ValueError("INPUT_PATH environment variable is required.")
        if not self.prompts:
            raise ValueError("OBJECT environment variable is required (comma-separated list).")
        if not self.is_new_dataset and not self.previous_dataset_path:
            raise ValueError("PREVIOUS_DATASET_PATH is required when IS_NEW_DATASET=false.")
        split_total = self.train_split + self.val_split + self.test_split
        if not (0.99 <= split_total <= 1.01):
            raise ValueError(f"TRAIN_SPLIT + VAL_SPLIT + TEST_SPLIT must sum to 1.0, got {split_total}.")


config = Config()
