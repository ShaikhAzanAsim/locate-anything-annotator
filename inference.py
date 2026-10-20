"""LocateAnything inference only.

Loads nvidia/LocateAnything-3B once, runs grounding on a single image against a
list of text prompts, and returns structured detections. Does not touch the
filesystem or know anything about YOLO/dataset layout -- that is
`core/service/annotator_service.py`'s job.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor, AutoTokenizer

from core.base_config import config
from utils.services.weight_dowload import download_model_if_needed

logger = logging.getLogger(__name__)

_BOX_PATTERN = re.compile(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>")


@dataclass
class Detection:
    """A single grounded detection in pixel coordinates.

    Attributes:
        class_name: The prompt/category this box was grounded against.
        confidence: Always 1.0 -- LocateAnything-3B does not emit a calibrated
            per-box confidence score (it generates <box> tokens via text
            generation, not objectness/class logits).
        bbox: (x1, y1, x2, y2) in pixel coordinates.
    """

    class_name: str
    confidence: float
    bbox: tuple[float, float, float, float]


class LocateAnythingModel:
    """Singleton wrapper around nvidia/LocateAnything-3B, loaded once per process."""

    _instance: "LocateAnythingModel | None" = None

    def __init__(self) -> None:
        self.device = config.device
        self.dtype = torch.bfloat16 if self.device.startswith("cuda") else torch.float32
        model_path = download_model_if_needed(config.model_local_dir, repo_id=config.model_name)
        logger.info("Loading LocateAnything model %s (device=%s)", model_path, self.device)
        self.model = (
            AutoModel.from_pretrained(model_path, trust_remote_code=True, dtype=self.dtype)
            .to(self.device)
            .eval()
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        logger.info("LocateAnything model loaded.")

    @classmethod
    def instance(cls) -> "LocateAnythingModel":
        """Return the process-wide model instance, loading it on first call."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @torch.inference_mode()
    def detect(self, image: Image.Image, prompts: list[str]) -> list[Detection]:
        """Ground each prompt against the image and collect all detections.

        The model has no way to tell us which box belongs to which category when
        several categories are packed into one prompt, so each prompt is
        grounded with its own single-phrase query and the label is attached
        from the query itself.

        Args:
            image: RGB image to run detection on.
            prompts: Category names / phrases to locate.

        Returns:
            All detections found across all prompts. A prompt that fails to
            generate (model error) is logged and skipped rather than raised,
            so one bad prompt/frame can't abort the whole batch job.
        """
        detections: list[Detection] = []
        for prompt in prompts:
            try:
                answer = self._generate(image, prompt)
            except Exception:
                logger.exception("LocateAnything inference failed for prompt %r, skipping.", prompt)
                continue
            for box in self._parse_boxes(answer, image.width, image.height):
                detections.append(Detection(class_name=prompt, confidence=1.0, bbox=box))
        return detections

    def _generate(self, image: Image.Image, prompt_text: str) -> str:
        prompt = config.prompt_template.format(categories=prompt_text)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = self.processor.py_apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos = self.processor.process_vision_info(messages)
        inputs = self.processor(text=[text], images=images, videos=videos, return_tensors="pt").to(self.device)

        output = self.model.generate(
            pixel_values=inputs["pixel_values"].to(self.dtype),
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            image_grid_hws=inputs.get("image_grid_hws"),
            tokenizer=self.tokenizer,
            max_new_tokens=config.max_new_tokens,
            use_cache=True,
            generation_mode=config.generation_mode,
            temperature=config.generation_temperature,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1,
        )
        # LocateAnything's generate() override (source: modeling_locateanything.py) already
        # decodes internally (skip_special_tokens=False, so <box>/<ref>/coordinate tokens survive
        # as literal text) and returns response[0] as a plain str when verbose is unset/False, or
        # (response[0], sampling_history, out_info) when verbose=True. We never pass verbose, so
        # `output` is already the answer string -- do not index into it.
        return output[0] if isinstance(output, tuple) else output

    @staticmethod
    def _parse_boxes(answer: str, width: int, height: int) -> list[tuple[float, float, float, float]]:
        boxes = []
        for match in _BOX_PATTERN.finditer(answer):
            x1, y1, x2, y2 = (int(g) for g in match.groups())
            boxes.append(
                (
                    x1 / 1000 * width,
                    y1 / 1000 * height,
                    x2 / 1000 * width,
                    y2 / 1000 * height,
                )
            )
        return boxes


def infer(image: Image.Image, prompts: list[str]) -> list[Detection]:
    """Simple functional entry point: run LocateAnything on one image.

    Args:
        image: RGB image to run detection on.
        prompts: Category names / phrases to locate.

    Returns:
        List of `Detection` found across all prompts.
    """
    return LocateAnythingModel.instance().detect(image, prompts)
