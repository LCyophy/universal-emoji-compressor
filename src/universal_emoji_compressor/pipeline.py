"""正式入口：原图 OCR 建索引，最终编码后的 160px OCR 作为可读性基准。"""

from __future__ import annotations

import io
import time
from collections import Counter

from PIL import Image

from . import core
from .gif_transparency_fix import quantize_rgba

BASE_PROCESS_CHUNK = core.process_chunk
VALIDATION_WEBP_QUALITY = 68
VALIDATION_GIF_COLORS = 96
BASELINE_MIN_SCORE = 0.55
CANDIDATE_MIN_SCORE = 0.35
MIN_BASELINE_CHARS = 2
MIN_CANDIDATE_PRECISION = 0.65
MIN_TEXT_RETENTION = 0.80


def validation_roundtrip(image: Image.Image, item: core.WorkItem) -> Image.Image:
    """近似最终编码后再解码，避免用无损缩略图高估文字可读性。"""
    if item.info.animated:
        return quantize_rgba(image, VALIDATION_GIF_COLORS).convert("RGBA")
    buffer = io.BytesIO()
    image.save(buffer, "WEBP", quality=VALIDATION_WEBP_QUALITY, method=6, exact=True)
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        return decoded.convert("RGBA").copy()


def text_retention(source_texts: list[str], candidate_texts: list[str]) -> float:
    """按整图字符多重集合计算召回，容忍换行和标点变化并抑制误识别。"""
    source = core.normalize_text("".join(source_texts))
    candidate = core.normalize_text("".join(candidate_texts))
    if not source:
        return 1.0
    if not candidate:
        return 0.0
    common = sum((Counter(source) & Counter(candidate)).values())
    recall = common / len(source)
    precision = common / len(candidate)
    if recall >= 1.0:
        return 1.0
    return recall if precision >= MIN_CANDIDATE_PRECISION else 0.0


def next_size(current: int, long_edge: int) -> int:
    for candidate in core.SIZES:
        if candidate > current:
            return min(candidate, long_edge)
    return min(core.SIZES[-1], long_edge)


def _recognized_texts(results: list[dict], min_score: float) -> list[str]:
    texts: list[str] = []
    known: set[str] = set()
    for row in results:
        text = str(row.get("text", "")).strip()
        normalized = core.normalize_text(text)
        if float(row.get("score", 0)) < min_score or not normalized:
            continue
        if normalized not in known:
            texts.append(text)
            known.add(normalized)
    return texts


def _reset_to_complexity(item: core.WorkItem) -> None:
    """160px 无可靠文字时，原图 OCR 仅留作索引，不再干预尺寸。"""
    source_constraints = list(item.decision.get("text_constraints") or [])
    selected, decision = core.choose_size(
        item.metrics, [], (item.info.width, item.info.height), 0
    )
    decision["reason"] = "complexity_no_160_ocr"
    if source_constraints:
        decision["source_ocr_constraints"] = source_constraints
    item.target_size = selected
    item.decision = decision


def validated_process_chunk(
    items: list[core.WorkItem],
    ocr: core.PaddleOCRBatch | None,
    vision: core.QwenVisionBatch | None,
    ocr_batch_size: int,
    vision_batch_size: int,
    min_text_px: float,
) -> dict[int, dict[str, float]]:
    timings = BASE_PROCESS_CHUNK(
        items, ocr, vision, ocr_batch_size, vision_batch_size, min_text_px
    )
    if ocr is None:
        return timings

    baseline_images: list[Image.Image] = []
    baseline_owners: list[core.WorkItem] = []
    for item in items:
        dimensions = core.output_dimensions(item.info.width, item.info.height, core.SIZES[-1])
        for frame in item.frames:
            resized = frame if frame.size == dimensions else frame.resize(
                dimensions, Image.Resampling.LANCZOS
            )
            baseline_images.append(validation_roundtrip(resized, item))
            baseline_owners.append(item)

    started = time.perf_counter()
    baseline_results = ocr.run(baseline_images, ocr_batch_size)
    elapsed = time.perf_counter() - started
    baseline_by_asset: dict[int, list[str]] = {item.asset_id: [] for item in items}
    for owner, result in zip(baseline_owners, baseline_results, strict=True):
        combined = baseline_by_asset[owner.asset_id] + _recognized_texts(
            result, BASELINE_MIN_SCORE
        )
        baseline_by_asset[owner.asset_id] = _recognized_texts(
            [{"text": text, "score": 1.0} for text in combined], BASELINE_MIN_SCORE
        )
    for item in items:
        timings[item.asset_id]["ocr"] += elapsed / max(1, len(items))
        baseline_by_asset[item.asset_id] = [
            text for text in baseline_by_asset[item.asset_id]
            if len(core.normalize_text(text)) >= MIN_BASELINE_CHARS
        ]
        if baseline_by_asset[item.asset_id]:
            item.target_size = min(core.SIZES[0], max(item.info.width, item.info.height))
            item.decision["selected"] = item.target_size
            item.decision["reason"] = "ocr_first_160_baseline"
            item.decision["ocr_160_baseline"] = baseline_by_asset[item.asset_id]
        else:
            _reset_to_complexity(item)

    unresolved = [item for item in items if baseline_by_asset[item.asset_id]]
    while unresolved:
        images: list[Image.Image] = []
        owners: list[core.WorkItem] = []
        for item in unresolved:
            dimensions = core.output_dimensions(
                item.info.width, item.info.height, item.target_size
            )
            for frame in item.frames:
                resized = frame if frame.size == dimensions else frame.resize(
                    dimensions, Image.Resampling.LANCZOS
                )
                images.append(validation_roundtrip(resized, item))
                owners.append(item)

        started = time.perf_counter()
        results = ocr.run(images, ocr_batch_size)
        elapsed = time.perf_counter() - started
        texts_by_asset: dict[int, list[str]] = {item.asset_id: [] for item in unresolved}
        for owner, result in zip(owners, results, strict=True):
            texts_by_asset[owner.asset_id].extend(
                _recognized_texts(result, CANDIDATE_MIN_SCORE)
            )

        retry: list[core.WorkItem] = []
        for item in unresolved:
            timings[item.asset_id]["ocr"] += elapsed / max(1, len(unresolved))
            recognized = texts_by_asset[item.asset_id]
            retention = text_retention(baseline_by_asset[item.asset_id], recognized)
            item.decision.setdefault("readability_validation", []).append(
                {
                    "size": item.target_size,
                    "retention": round(retention, 4),
                    "recognized": recognized,
                }
            )
            candidate = next_size(item.target_size, max(item.info.width, item.info.height))
            if retention < MIN_TEXT_RETENTION and candidate > item.target_size:
                item.target_size = candidate
                item.decision["selected"] = candidate
                item.decision["reason"] = "post_resize_ocr"
                if candidate >= core.SIZES[-1]:
                    item.decision["readability_validation"].append(
                        {
                            "size": candidate,
                            "retention": 1.0,
                            "recognized": baseline_by_asset[item.asset_id],
                            "baseline": True,
                        }
                    )
                else:
                    retry.append(item)
        unresolved = retry
    return timings


core.process_chunk = validated_process_chunk

if __name__ == "__main__":
    core.main()
