"""Reads INPUT_PATH and yields (name, image) pairs.

Supports a single image, a single video, a folder of images, a folder of
videos (or a mix). Also accepts http(s) URLs to a single image or a single
video. For URLs: images are loaded into PIL from bytes; videos are streamed
into a temporary file and fed to OpenCV.

Runs no inference and writes nothing to disk -- that is `annotator_service.py`'s
job.
"""
from __future__ import annotations

import io
import logging
import mimetypes
import shutil
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Iterator

import cv2
from PIL import Image, UnidentifiedImageError

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def _is_url(path: str) -> bool:
    lower = path.lower()
    return lower.startswith("http://") or lower.startswith("https://")


def iter_source_images(input_path: str, frame_skip: int = 30, max_frames: int = 0) -> Iterator[tuple[str, Image.Image]]:
    """Yield (name, image) pairs from INPUT_PATH.

    Args:
        input_path: A single image file, a single video file, a folder
            containing any mix of image/video files, or an http(s) URL to a
            single image or single video.
        frame_skip: Sample every Nth frame from each video.
        max_frames: Cap on frames extracted per video. 0 means no cap.

    Yields:
        (deterministic_name, PIL.Image) pairs. Corrupt/unreadable files are
        logged and skipped rather than raised.

    Raises:
        FileNotFoundError: If a local input_path does not exist.
    """
    if _is_url(input_path):
        yield from _iter_url(input_path, frame_skip, max_frames)
        return

    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"INPUT_PATH does not exist: {input_path}")

    if path.is_file():
        yield from _iter_file(path, frame_skip, max_frames)
    else:
        for child in sorted(p for p in path.iterdir() if p.is_file()):
            yield from _iter_file(child, frame_skip, max_frames)


def _iter_file(path: Path, frame_skip: int, max_frames: int) -> Iterator[tuple[str, Image.Image]]:
    suffix = path.suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        yield from _load_image(path)
    elif suffix in VIDEO_EXTENSIONS:
        yield from _iter_video_frames(path, frame_skip, max_frames)
    else:
        logger.warning("Skipping unsupported file type: %s", path)


def _load_image(path: Path) -> Iterator[tuple[str, Image.Image]]:
    try:
        yield path.stem, Image.open(path).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        logger.warning("Skipping corrupt image %s: %s", path, exc)


def _iter_video_frames(path: Path, frame_skip: int, max_frames: int) -> Iterator[tuple[str, Image.Image]]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        logger.warning("Skipping unreadable/corrupt video: %s", path)
        return

    frame_index = 0
    extracted = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % frame_skip == 0:
                extracted += 1
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                yield f"{path.stem}_frame_{extracted:06d}", Image.fromarray(rgb)
                if max_frames and extracted >= max_frames:
                    break
            frame_index += 1
    except cv2.error as exc:
        logger.warning("Stopped reading %s after %d frames due to decode error: %s", path, extracted, exc)
    finally:
        capture.release()


def _iter_url(url: str, frame_skip: int, max_frames: int) -> Iterator[tuple[str, Image.Image]]:
    """Handle a single http(s) URL pointing to an image or a video."""
    try:
        resp = urllib.request.urlopen(url)
    except urllib.error.URLError as exc:
        logger.warning("Skipping URL %s: %s", url, exc)
        return

    # Prefer Content-Type header when available
    content_type = resp.headers.get("Content-Type", "").split(";")[0].strip().lower()
    path_ext = Path(urllib.parse.urlparse(url).path).suffix.lower()

    # Decide whether this is an image or video
    is_image = False
    is_video = False
    if content_type.startswith("image/"):
        is_image = True
    elif content_type.startswith("video/"):
        is_video = True
    else:
        if path_ext in IMAGE_EXTENSIONS:
            is_image = True
        elif path_ext in VIDEO_EXTENSIONS:
            is_video = True

    if is_image:
        try:
            data = resp.read()
            img = Image.open(io.BytesIO(data)).convert("RGB")
            name = Path(urllib.parse.urlparse(url).path).stem or "remote_image"
            yield name, img
        except (UnidentifiedImageError, OSError) as exc:
            logger.warning("Skipping unreadable image URL %s: %s", url, exc)
        return

    if is_video:
        # Write to a temp file and stream into the existing video iterator, then cleanup.
        suffix = path_ext if path_ext else mimetypes.guess_extension(content_type) or ".mp4"
        tmp = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
                tmp = f.name
                # Stream copy
                shutil.copyfileobj(resp, f)

            for item in _iter_video_frames(Path(tmp), frame_skip, max_frames):
                yield item
        except Exception as exc:
            logger.warning("Failed processing video URL %s: %s", url, exc)
        finally:
            if tmp and Path(tmp).exists():
                try:
                    Path(tmp).unlink()
                except OSError:
                    logger.debug("Failed to remove temp file %s", tmp)
        return

    logger.warning("Could not determine content type for URL %s (Content-Type=%s). Skipping.", url, content_type)
