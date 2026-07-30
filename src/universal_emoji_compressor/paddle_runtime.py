"""隔离 PaddleOCR：阻止 ModelScope 在 Paddle 进程中顺带加载 PyTorch/cuDNN。"""

from __future__ import annotations

import importlib.util
import json
import os
from typing import Any

import numpy as np
from PIL import Image

from . import core


class PaddleOCRBatch(core.PaddleOCRBatch):
    def __init__(self, device: str) -> None:
        core.configure_windows_cuda_dlls()
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

        original_find_spec = importlib.util.find_spec

        def isolated_find_spec(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "torch" or name.startswith("torch."):
                return None
            return original_find_spec(name, *args, **kwargs)

        importlib.util.find_spec = isolated_find_spec
        try:
            from paddleocr import PaddleOCR
        finally:
            importlib.util.find_spec = original_find_spec
        self.engine = PaddleOCR(
            device=device,
            text_detection_model_name="PP-OCRv5_mobile_det",
            text_recognition_model_name="PP-OCRv5_mobile_rec",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )

    @staticmethod
    def _parse(result: Any) -> list[dict[str, Any]]:
        payload = getattr(result, "json", result)
        if callable(payload):
            payload = payload()
        if isinstance(payload, str):
            payload = json.loads(payload)
        payload = payload.get("res", payload)
        return [
            {
                "text": str(text).strip(),
                "score": float(score),
                "box": [[float(x), float(y)] for x, y in box],
            }
            for text, score, box in zip(
                payload.get("rec_texts", []),
                payload.get("rec_scores", []),
                payload.get("rec_polys", payload.get("dt_polys", [])),
            )
            if str(text).strip()
        ]

    def run(self, images: list[Image.Image], batch_size: int) -> list[list[dict[str, Any]]]:
        results: list[list[dict[str, Any]]] = []
        for start in range(0, len(images), batch_size):
            arrays = [
                np.asarray(image.convert("RGB"))
                for image in images[start : start + batch_size]
            ]
            results.extend(self._parse(result) for result in self.engine.predict(arrays))
        return results


def install() -> None:
    core.PaddleOCRBatch = PaddleOCRBatch
