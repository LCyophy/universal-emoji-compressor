"""正式入口：复杂度预测后，以缩放后 OCR 保留率做闭环升档。"""

from __future__ import annotations

import time
from difflib import SequenceMatcher

from PIL import Image

from . import core

BASE_PROCESS_CHUNK = core.process_chunk


def matches(source: str, candidates: list[str]) -> bool:
    source = core.normalize_text(source)
    if not source:
        return True
    for candidate in candidates:
        candidate = core.normalize_text(candidate)
        if not candidate:
            continue
        if (
            (source in candidate or candidate in source)
            and min(len(source), len(candidate)) >= max(1, round(len(source) * 0.55))
        ):
            return True
        if SequenceMatcher(None, source, candidate).ratio() >= 0.76:
            return True
    return False


def next_size(current: int, long_edge: int) -> int:
    for candidate in core.SIZES:
        if candidate > current:
            return min(candidate, long_edge)
    return min(core.SIZES[-1], long_edge)


def validated_process_chunk(
    items: list[core.WorkItem],
    ocr: core.PaddleOCRBatch | None,
    vision: core.QwenVisionBatch | None,
    ocr_batch_size: int,
    vision_batch_size: int,
    min_text_px: float,
) -> dict[int, dict[str, float]]:
    timings = BASE_PROCESS_CHUNK(
        items,
        ocr,
        vision,
        ocr_batch_size,
        vision_batch_size,
        min_text_px,
    )
    if ocr is None:
        return timings

    unresolved = [
        item
        for item in items
        if any(float(group.get("best_score", 0)) >= 0.55 for group in (item.ocr_merged or []))
    ]
    while unresolved:
        images: list[Image.Image] = []
        owners: list[core.WorkItem] = []
        for item in unresolved:
            dimensions = core.output_dimensions(item.info.width, item.info.height, item.target_size)
            for frame in item.frames:
                resized = frame if frame.size == dimensions else frame.resize(dimensions, Image.Resampling.LANCZOS)
                images.append(resized)
                owners.append(item)

        started = time.perf_counter()
        results = ocr.run(images, ocr_batch_size)
        elapsed = time.perf_counter() - started
        texts_by_asset: dict[int, list[str]] = {item.asset_id: [] for item in unresolved}
        for owner, result in zip(owners, results):
            texts_by_asset[owner.asset_id].extend(
                row["text"] for row in result if float(row.get("score", 0)) >= 0.35
            )

        retry: list[core.WorkItem] = []
        for item in unresolved:
            timings[item.asset_id]["ocr"] += elapsed / max(1, len(unresolved))
            source_groups = [
                group
                for group in (item.ocr_merged or [])
                if float(group.get("best_score", 0)) >= 0.55
            ]
            retained = sum(matches(group["text"], texts_by_asset[item.asset_id]) for group in source_groups)
            retention = retained / max(1, len(source_groups))
            item.decision.setdefault("readability_validation", []).append(
                {
                    "size": item.target_size,
                    "retention": round(retention, 4),
                    "recognized": texts_by_asset[item.asset_id],
                }
            )
            long_edge = max(item.info.width, item.info.height)
            candidate = next_size(item.target_size, long_edge)
            if retention < 0.80 and candidate > item.target_size:
                item.target_size = candidate
                item.decision["selected"] = candidate
                item.decision["reason"] = "post_resize_ocr"
                retry.append(item)
        unresolved = retry
    return timings


core.process_chunk = validated_process_chunk

if __name__ == "__main__":
    core.main()
