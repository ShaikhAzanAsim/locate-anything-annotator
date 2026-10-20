"""YOLO training only.

Takes a ready data.yaml, reads every hyperparameter from environment
variables (via core.base_config), runs Ultralytics YOLO training, and copies
the resulting weights out to WEIGHTS_DIR. No dataset-generation or
annotation logic belongs here -- see core/service/annotator_service.py and
utils/services/split_dataset_service.py for that.
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path

from ultralytics import YOLO

from core.base_config import config

logger = logging.getLogger(__name__)


def run_training(data_yaml_path: Path) -> Path:
    """Train a YOLO model on data_yaml_path and export its weights.

    Args:
        data_yaml_path: Path to the Ultralytics data.yaml to train against.

    Returns:
        Path to the best checkpoint copied into WEIGHTS_DIR.
    """
    logger.info("Starting YOLO training: base_model=%s epochs=%d", config.yolo_model, config.epochs)
    model = YOLO(config.yolo_model)
    model.train(
        data=str(data_yaml_path),
        epochs=config.epochs,
        imgsz=config.image_size,
        batch=config.batch,
        device=config.device,
        workers=config.workers,
        save_period=config.save_period,
        project=config.project_name,
        name=config.run_name,
        **config.extra_train_args,
    )
    return _export_weights(Path(model.trainer.save_dir))


def _export_weights(run_dir: Path) -> Path:
    """Copy best.pt/last.pt out of the (nested) run directory into WEIGHTS_DIR."""
    weights_dir = Path(config.weights_dir)
    weights_dir.mkdir(parents=True, exist_ok=True)

    best_src = run_dir / "weights" / "best.pt"
    last_src = run_dir / "weights" / "last.pt"
    best_dst = weights_dir / "best.pt"
    if best_src.exists():
        shutil.copy2(best_src, best_dst)
    if last_src.exists():
        shutil.copy2(last_src, weights_dir / "last.pt")

    logger.info("Weights exported to %s", weights_dir)
    return best_dst if best_src.exists() else last_src


if __name__ == "__main__":
    # Standalone re-training against an already-built dataset, bypassing
    # dataset generation entirely (main.py is the normal full-pipeline entrypoint).
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run_training(Path(config.output_dataset_path) / "data.yaml")
