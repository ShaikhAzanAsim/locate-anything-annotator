import logging
from pathlib import Path

from huggingface_hub import snapshot_download

logger = logging.getLogger(__name__)

MODEL_ID = "nvidia/LocateAnything-3B"


def download_model_if_needed(
    model_dir: str = "./weight/LocateAnything-3B",
    repo_id: str = MODEL_ID,
) -> str:
    """
    Downloads the LocateAnything model if it is not already present.
    Returns the local model directory.
    """

    model_path = Path(model_dir)

    # Enough to determine whether the model is already present
    index_file = model_path / "model.safetensors.index.json"

    if index_file.exists():
        logger.info("LocateAnything weights already present at %s", model_path)
        return str(model_path)

    logger.info("Downloading LocateAnything weights (%s) to %s", repo_id, model_path)

    snapshot_download(
        repo_id=repo_id,
        local_dir=str(model_path),
        local_dir_use_symlinks=False,   # Safe for Docker/Windows
        resume_download=True,
    )

    logger.info("Download completed.")

    return str(model_path)