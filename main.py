"""Container entrypoint.

Batch job, not a service: reads configuration, generates a YOLO dataset from
INPUT_PATH via LocateAnything, splits/merges it, trains YOLO, and exits. Runs
once per container start -- intended to back a Kubernetes Job, not a
Deployment.
"""
from __future__ import annotations

import logging

from core.base_config import config
from core.service.annotator_service import generate_dataset
from train import run_training
from utils.services.dataset_visualizer_service import visualize_dataset
from utils.services.split_dataset_service import build_final_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    logger.info("Loading configuration...")
    config.validate()

    logger.info("Loading LocateAnything model and extracting/annotating frames...")
    generate_dataset()

    logger.info("Splitting dataset...")
    data_yaml_path = build_final_dataset()

    if config.visualize_dataset:
        logger.info("Rendering label preview images...")
        visualize_dataset(config.output_dataset_path, config.visualization_dir)

    logger.info("Starting YOLO training...")
    best_weights = run_training(data_yaml_path)
    logger.info("Training complete. Best weights: %s", best_weights)

    logger.info("Exiting.")


if __name__ == "__main__":
    main()
