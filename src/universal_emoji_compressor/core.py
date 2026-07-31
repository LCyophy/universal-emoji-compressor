from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import re
import sqlite3
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageSequence, UnidentifiedImageError
from tqdm import tqdm

LOGGER = logging.getLogger("emoji-pipeline")
SIZES = (64, 96, 128, 160)
SUFFIXES = {".jpg", ".jpeg", ".jfif", ".png", ".webp", ".gif", ".bmp", ".tif", ".tiff", ".avif"}
_DLL_HANDLES: list[Any] = []


def configure_windows_cuda_dlls() -> None:
    """让 Windows 能找到随 wheel 安装的 CUDA/cuDNN DLL。"""
    if os.name != "nt" or not hasattr(os, "add_dll_directory"):
        return
    site_packages = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
    if not site_packages.exists():
        return
    directories = list(site_packages.glob("*/bin"))
    current_path = os.environ.get("PATH", "")
    dll_path = os.pathsep.join(str(directory) for directory in directories)
    if dll_path:
        os.environ["PATH"] = dll_path + os.pathsep + current_path
    for directory in directories:
        try:
            _DLL_HANDLES.append(os.add_dll_directory(str(directory)))
        except OSError:
            pass


@dataclass(slots=True)
class MediaInfo:
    format: str
    width: int
    height: int
    frame_count: int
    durations: list[int]
    loop: int
    disposals: list[int]

    @property
    def animated(self) -> bool:
        return self.frame_count > 1

    @property
    def duration_ms(self) -> int:
        return sum(self.durations)


@dataclass(slots=True)
class VisualMetrics:
    entropy: float
    edge_density: float
    effective_colors: float
    colorfulness: float
    subject_occupancy: float
    face_count: int
    small_detail_ratio: float
    score: float


@dataclass(slots=True)
class WorkItem:
    path: Path
    relative_path: str
    sha256: str
    asset_id: int
    info: MediaInfo
    frame_indices: list[int]
    frames: list[Image.Image]
    metrics: VisualMetrics | None = None
    ocr_by_frame: list[dict[str, Any]] | None = None
    ocr_merged: list[dict[str, Any]] | None = None
    target_size: int = 160
    decision: dict[str, Any] | None = None
    understanding: dict[str, Any] | None = None


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _frame_value(value: Any, index: int, default: int) -> int:
    """读取 Pillow 可能以标量或逐帧列表返回的 GIF 元数据。"""
    if isinstance(value, (list, tuple)):
        value = value[min(index, len(value) - 1)] if value else default
    return int(default if value is None else value)


def inspect_media(path: Path) -> MediaInfo:
    with Image.open(path) as image:
        frame_count = int(getattr(image, "n_frames", 1))
        durations: list[int] = []
        disposals: list[int] = []
        for index in range(frame_count):
            image.seek(index)
            durations.append(_frame_value(image.info.get("duration", 100), index, 100))
            disposals.append(_frame_value(getattr(image, "disposal_method", 2), index, 2))
        return MediaInfo(
            format=(image.format or path.suffix.lstrip(".")).upper(),
            width=image.width,
            height=image.height,
            frame_count=frame_count,
            durations=durations,
            loop=int(image.info.get("loop", 0)),
            disposals=disposals,
        )


def _uniform_indices(frame_count: int, count: int) -> set[int]:
    if frame_count <= 1:
        return {0}
    count = min(frame_count, max(2, count))
    return {round(i * (frame_count - 1) / (count - 1)) for i in range(count)}


def load_keyframes(path: Path, limit: int) -> tuple[list[int], list[Image.Image]]:
    """均匀采样 + 画面突变帧，减少 GIF 文字漏检。"""
    with Image.open(path) as image:
        frame_count = int(getattr(image, "n_frames", 1))
        if frame_count == 1:
            return [0], [image.convert("RGBA").copy()]

        thumbnails: list[np.ndarray] = []
        for index in range(frame_count):
            image.seek(index)
            thumb = image.convert("L")
            thumb.thumbnail((64, 64), Image.Resampling.BILINEAR)
            canvas = Image.new("L", (64, 64))
            canvas.paste(thumb, ((64 - thumb.width) // 2, (64 - thumb.height) // 2))
            thumbnails.append(np.asarray(canvas, dtype=np.float32))

        differences = [
            (float(np.mean(np.abs(thumbnails[index] - thumbnails[index - 1]))), index)
            for index in range(1, frame_count)
        ]
        selected = _uniform_indices(frame_count, max(2, math.ceil(limit * 0.65)))
        for _, index in sorted(differences, reverse=True):
            if len(selected) >= min(limit, frame_count):
                break
            selected.add(index)

    indices = sorted(selected)
    frames: list[Image.Image] = []
    with Image.open(path) as image:
        for index in indices:
            image.seek(index)
            frames.append(image.convert("RGBA").copy())
    return indices, frames


def _entropy(values: np.ndarray, bins: int = 256) -> float:
    histogram = np.histogram(values, bins=bins, range=(0, 256))[0].astype(np.float64)
    probability = histogram[histogram > 0] / histogram.sum()
    return float(-(probability * np.log2(probability)).sum())


def visual_metrics(image: Image.Image) -> VisualMetrics:
    rgba = image.convert("RGBA")
    rgba.thumbnail((320, 320), Image.Resampling.LANCZOS)
    array = np.asarray(rgba, dtype=np.uint8)
    alpha = array[:, :, 3]
    rgb = array[:, :, :3]
    white = np.full_like(rgb, 255)
    alpha_f = alpha[:, :, None].astype(np.float32) / 255.0
    composite = (rgb * alpha_f + white * (1.0 - alpha_f)).astype(np.uint8)
    gray = cv2.cvtColor(composite, cv2.COLOR_RGB2GRAY)

    entropy = _entropy(gray) / 8.0
    edges = cv2.Canny(gray, 50, 140)
    edge_density = float(np.mean(edges > 0))

    quantized = (composite // 32).reshape(-1, 3)
    _, counts = np.unique(quantized, axis=0, return_counts=True)
    probabilities = counts / counts.sum()
    effective = float(np.exp(-(probabilities * np.log(probabilities)).sum()))
    effective_colors = float(np.clip(math.log2(max(effective, 1)) / 9.0, 0, 1))

    pixels = composite.astype(np.float32)
    rg = pixels[:, :, 0] - pixels[:, :, 1]
    yb = 0.5 * (pixels[:, :, 0] + pixels[:, :, 1]) - pixels[:, :, 2]
    colorfulness = float(
        np.clip(
            (math.sqrt(float(rg.std() ** 2 + yb.std() ** 2))
             + 0.3 * math.sqrt(float(rg.mean() ** 2 + yb.mean() ** 2))) / 128.0,
            0,
            1,
        )
    )

    if np.any(alpha < 250):
        foreground = alpha > 24
    else:
        border = np.concatenate((composite[0], composite[-1], composite[:, 0], composite[:, -1]), axis=0)
        background = np.median(border.astype(np.float32), axis=0)
        distance = np.linalg.norm(composite.astype(np.float32) - background, axis=2)
        threshold = max(22.0, float(np.quantile(distance, 0.62)))
        foreground = distance > threshold
        foreground = cv2.morphologyEx(
            foreground.astype(np.uint8),
            cv2.MORPH_CLOSE,
            np.ones((5, 5), dtype=np.uint8),
        ) > 0
    subject_occupancy = float(np.mean(foreground))

    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(str(cascade_path))
    min_side = max(18, min(gray.shape) // 12)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.12, minNeighbors=4, minSize=(min_side, min_side))
    face_count = len(faces)

    components, _, stats, _ = cv2.connectedComponentsWithStats((edges > 0).astype(np.uint8), 8)
    image_area = gray.shape[0] * gray.shape[1]
    small = 0
    for component in range(1, components):
        area = int(stats[component, cv2.CC_STAT_AREA])
        if 2 <= area <= max(3, image_area * 0.004):
            small += area
    small_detail_ratio = float(np.clip(small / max(1, image_area) * 8.0, 0, 1))

    occupancy_interest = 1.0 - min(1.0, abs(subject_occupancy - 0.45) / 0.55)
    face_interest = min(face_count, 3) / 3.0
    score = float(
        np.clip(
            0.27 * entropy
            + 0.22 * edge_density
            + 0.13 * effective_colors
            + 0.10 * colorfulness
            + 0.11 * occupancy_interest
            + 0.09 * face_interest
            + 0.08 * small_detail_ratio,
            0,
            1,
        )
    )
    return VisualMetrics(
        entropy=entropy,
        edge_density=edge_density,
        effective_colors=effective_colors,
        colorfulness=colorfulness,
        subject_occupancy=subject_occupancy,
        face_count=face_count,
        small_detail_ratio=small_detail_ratio,
        score=score,
    )


def aggregate_metrics(values: list[VisualMetrics]) -> VisualMetrics:
    if not values:
        return VisualMetrics(0, 0, 0, 0, 0, 0, 0, 0)
    quantile_fields = (
        "entropy",
        "edge_density",
        "effective_colors",
        "colorfulness",
        "subject_occupancy",
        "small_detail_ratio",
        "score",
    )
    aggregated = {
        key: float(np.quantile([getattr(item, key) for item in values], 0.80))
        for key in quantile_fields
    }
    return VisualMetrics(face_count=max(item.face_count for item in values), **aggregated)


def normalize_text(text: str) -> str:
    return re.sub(r"[\W_]+", "", text, flags=re.UNICODE).lower()


def merge_frame_ocr(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """跨帧模糊合并同一句文字，同时保留出现帧，便于检查漏检。"""
    groups: list[dict[str, Any]] = []
    for item in sorted(items, key=lambda row: (row["frame_index"], -row["score"])):
        normalized = normalize_text(item["text"])
        if not normalized:
            continue
        match = None
        for group in groups:
            ratio = SequenceMatcher(None, normalized, group["normalized"]).ratio()
            if ratio >= 0.82 or normalized in group["normalized"] or group["normalized"] in normalized:
                match = group
                break
        if match is None:
            groups.append(
                {
                    "text": item["text"],
                    "normalized": normalized,
                    "best_score": item["score"],
                    "box": item["box"],
                    "frames": [item["frame_index"]],
                    "variants": [item["text"]],
                }
            )
        else:
            match["frames"].append(item["frame_index"])
            if item["text"] not in match["variants"]:
                match["variants"].append(item["text"])
            if item["score"] > match["best_score"]:
                match.update(text=item["text"], normalized=normalized, best_score=item["score"], box=item["box"])
    for group in groups:
        group["frames"] = sorted(set(group["frames"]))
    return groups


def _box_character_height(item: dict[str, Any]) -> float:
    box = item.get("box") or []
    if len(box) < 4:
        return 0.0
    xs = [float(point[0]) for point in box]
    ys = [float(point[1]) for point in box]
    width = max(xs) - min(xs)
    height = max(ys) - min(ys)
    characters = max(1, len(normalize_text(str(item.get("text", "")))))
    return min(height, width / characters)


def choose_size(
    metrics: VisualMetrics,
    ocr_groups: list[dict[str, Any]],
    original_size: tuple[int, int],
    min_text_px: float,
) -> tuple[int, dict[str, Any]]:
    # 阈值偏向节省空间，随后由 OCR 可读性设置硬下限。
    if metrics.score < 0.235:
        visual_size = 64
    elif metrics.score < 0.335:
        visual_size = 96
    elif metrics.score < 0.455:
        visual_size = 128
    else:
        visual_size = 160

    text_size = 64
    text_constraints: list[dict[str, Any]] = []
    long_edge = max(original_size)
    for group in ocr_groups:
        if float(group["best_score"]) < 0.40:
            continue
        char_height = _box_character_height(group)
        required = 160
        for candidate in SIZES:
            if char_height * candidate / long_edge >= min_text_px:
                required = candidate
                break
        text_size = max(text_size, required)
        text_constraints.append(
            {"text": group["text"], "source_char_px": round(char_height, 2), "required": required}
        )
    selected = max(visual_size, text_size)
    selected = min(selected, long_edge)  # 不放大小图
    return selected, {
        "visual_size": visual_size,
        "text_size": text_size if text_constraints else None,
        "selected": selected,
        "reason": "text" if text_constraints and text_size > visual_size else "complexity",
        "text_constraints": text_constraints,
    }


def output_dimensions(width: int, height: int, target: int) -> tuple[int, int]:
    scale = min(1.0, target / max(width, height))
    return max(1, round(width * scale)), max(1, round(height * scale))


def encode_media(
    source: Path,
    output_base: Path,
    info: MediaInfo,
    dimensions: tuple[int, int],
    webp_quality: int,
    gif_colors: int,
) -> Path:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    if not info.animated:
        output = output_base.with_suffix(".webp")
        with Image.open(source) as image:
            frame = image.convert("RGBA")
            if frame.size != dimensions:
                frame = frame.resize(dimensions, Image.Resampling.LANCZOS)
            frame.save(output, "WEBP", quality=webp_quality, method=6, exact=True)
        return output

    if info.format == "GIF":
        output = output_base.with_suffix(".gif")
        frames: list[Image.Image] = []
        with Image.open(source) as image:
            for index in range(info.frame_count):
                image.seek(index)
                frame = image.convert("RGBA")
                if frame.size != dimensions:
                    frame = frame.resize(dimensions, Image.Resampling.LANCZOS)
                frames.append(frame.convert("P", palette=Image.Palette.ADAPTIVE, colors=gif_colors))
        frames[0].save(
            output,
            "GIF",
            save_all=True,
            append_images=frames[1:],
            duration=info.durations,
            loop=info.loop,
            disposal=info.disposals,
            optimize=True,
        )
        return output

    output = output_base.with_suffix(".webp")
    with Image.open(source) as image:
        frames = []
        for frame in ImageSequence.Iterator(image):
            rgba = frame.convert("RGBA")
            if rgba.size != dimensions:
                rgba = rgba.resize(dimensions, Image.Resampling.LANCZOS)
            frames.append(rgba)
    frames[0].save(
        output,
        "WEBP",
        save_all=True,
        append_images=frames[1:],
        duration=info.durations,
        loop=info.loop,
        quality=webp_quality,
        method=6,
        minimize_size=True,
        exact=True,
    )
    return output


class PaddleOCRBatch:
    def __init__(self, device: str) -> None:
        configure_windows_cuda_dlls()
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
        from paddleocr import PaddleOCR

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
        texts = payload.get("rec_texts", [])
        scores = payload.get("rec_scores", [])
        boxes = payload.get("rec_polys", payload.get("dt_polys", []))
        return [
            {
                "text": str(text).strip(),
                "score": float(score),
                "box": [[float(x), float(y)] for x, y in box],
            }
            for text, score, box in zip(texts, scores, boxes)
            if str(text).strip()
        ]

    def run(self, images: list[Image.Image], batch_size: int) -> list[list[dict[str, Any]]]:
        all_results: list[list[dict[str, Any]]] = []
        for start in range(0, len(images), batch_size):
            arrays = [np.asarray(image.convert("RGB")) for image in images[start : start + batch_size]]
            all_results.extend(self._parse(result) for result in self.engine.predict(arrays))
        return all_results


class QwenVisionBatch:
    def __init__(self, model_name: str, max_pixels: int) -> None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.torch = torch
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            device_map="cuda:0",
        )
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            min_pixels=128 * 28 * 28,
            max_pixels=max_pixels,
        )

    def run(
        self,
        images: list[Image.Image],
        ocr_texts: list[str],
        batch_size: int,
    ) -> list[dict[str, Any]]:
        outputs: list[dict[str, Any]] = []
        prompt_base = (
            "分析这个表情包，只输出合法JSON，字段为："
            "description（一句简体中文描述），emotion（情绪数组），"
            "objects（人物/动物/物体数组），tags（简体中文搜索标签数组），"
            "safety（normal/adult/violence）。识别文字："
        )
        for start in range(0, len(images), batch_size):
            batch_images = [image.convert("RGB") for image in images[start : start + batch_size]]
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt_base + (ocr_text or "无")},
                    ],
                }
                for image, ocr_text in zip(batch_images, ocr_texts[start : start + batch_size])
            ]
            texts = [
                self.processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
                for message in messages
            ]
            inputs = self.processor(
                text=texts,
                images=batch_images,
                padding=True,
                return_tensors="pt",
            ).to(self.model.device)
            with self.torch.inference_mode():
                generated = self.model.generate(**inputs, max_new_tokens=160, do_sample=False)
            trimmed = [
                output[len(source) :]
                for source, output in zip(inputs.input_ids, generated)
            ]
            decoded = self.processor.batch_decode(trimmed, skip_special_tokens=True)
            for text in decoded:
                try:
                    begin, end = text.index("{"), text.rindex("}") + 1
                    outputs.append(json.loads(text[begin:end]))
                except (ValueError, json.JSONDecodeError):
                    outputs.append(
                        {"description": text.strip(), "emotion": [], "objects": [], "tags": [], "safety": "unknown"}
                    )
        return outputs


SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;
CREATE TABLE IF NOT EXISTS assets (
 id INTEGER PRIMARY KEY AUTOINCREMENT,
 sha256 TEXT NOT NULL UNIQUE,
 output_path TEXT, source_format TEXT, output_format TEXT,
 width INTEGER, height INTEGER, output_width INTEGER, output_height INTEGER,
 frame_count INTEGER, duration_ms INTEGER, target_size INTEGER,
 metrics_json TEXT, decision_json TEXT, sampled_frames_json TEXT,
 ocr_text TEXT, ocr_json TEXT, understanding_json TEXT,
 source_bytes INTEGER, output_bytes INTEGER,
 ocr_seconds REAL, vision_seconds REAL, encode_seconds REAL,
 status TEXT NOT NULL DEFAULT 'pending', error TEXT,
 created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
 updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS sources (
 id INTEGER PRIMARY KEY AUTOINCREMENT,
 asset_id INTEGER NOT NULL REFERENCES assets(id),
 original_path TEXT NOT NULL UNIQUE,
 file_size INTEGER NOT NULL, mtime_ns INTEGER NOT NULL
);
CREATE VIRTUAL TABLE IF NOT EXISTS asset_search USING fts5(
 asset_id UNINDEXED, ocr_text, description, tags
);
"""


class Store:
    def __init__(self, output: Path) -> None:
        self.output = output
        self.connection = sqlite3.connect(output / "index.sqlite")
        self.connection.row_factory = sqlite3.Row
        self.connection.executescript(SCHEMA)

    def prepare(self, path: str, sha256: str, size: int, mtime_ns: int) -> tuple[int, bool]:
        row = self.connection.execute("SELECT id,status FROM assets WHERE sha256=?", (sha256,)).fetchone()
        if row:
            asset_id = int(row["id"])
            complete = row["status"] in {"ok", "ignored"}
        else:
            asset_id = int(self.connection.execute("INSERT INTO assets(sha256) VALUES(?)", (sha256,)).lastrowid)
            complete = False
        self.connection.execute(
            """
            INSERT INTO sources(asset_id,original_path,file_size,mtime_ns) VALUES(?,?,?,?)
            ON CONFLICT(original_path) DO UPDATE SET
             asset_id=excluded.asset_id,file_size=excluded.file_size,mtime_ns=excluded.mtime_ns
            """,
            (asset_id, path, size, mtime_ns),
        )
        self.connection.commit()
        return asset_id, complete

    def complete(self, item: WorkItem, output_path: Path, timings: dict[str, float]) -> None:
        info = inspect_media(output_path)
        understanding = item.understanding or {}
        ocr_text = " ".join(group["text"] for group in (item.ocr_merged or []))
        fields = {
            "output_path": output_path.relative_to(self.output).as_posix(),
            "source_format": item.info.format,
            "output_format": info.format,
            "width": item.info.width,
            "height": item.info.height,
            "output_width": info.width,
            "output_height": info.height,
            "frame_count": item.info.frame_count,
            "duration_ms": item.info.duration_ms,
            "target_size": item.target_size,
            "metrics_json": asdict(item.metrics) if item.metrics else {},
            "decision_json": item.decision or {},
            "sampled_frames_json": item.frame_indices,
            "ocr_text": ocr_text,
            "ocr_json": {"merged": item.ocr_merged or [], "by_frame": item.ocr_by_frame or []},
            "understanding_json": understanding,
            "source_bytes": item.path.stat().st_size,
            "output_bytes": output_path.stat().st_size,
            "ocr_seconds": timings.get("ocr", 0),
            "vision_seconds": timings.get("vision", 0),
            "encode_seconds": timings.get("encode", 0),
        }
        columns = list(fields)
        values = [json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value for value in fields.values()]
        self.connection.execute(
            f"UPDATE assets SET {','.join(f'{key}=?' for key in columns)},status='ok',error=NULL,"
            "updated_at=CURRENT_TIMESTAMP WHERE id=?",
            (*values, item.asset_id),
        )
        tags: list[str] = []
        for key in ("emotion", "objects", "tags"):
            value = understanding.get(key, [])
            if isinstance(value, list):
                tags.extend(map(str, value))
        self.connection.execute("DELETE FROM asset_search WHERE asset_id=?", (item.asset_id,))
        self.connection.execute(
            "INSERT INTO asset_search(asset_id,ocr_text,description,tags) VALUES(?,?,?,?)",
            (item.asset_id, ocr_text, understanding.get("description", ""), " ".join(tags)),
        )
        self.connection.commit()

    def ignore(self, asset_id: int, error: str) -> None:
        self.connection.execute(
            "UPDATE assets SET status='ignored',error=?,updated_at=CURRENT_TIMESTAMP WHERE id=?",
            (error[:4000], asset_id),
        )
        self.connection.commit()

    def fail(self, asset_id: int, error: str) -> None:
        self.connection.execute(
            "UPDATE assets SET status='error',error=?,updated_at=CURRENT_TIMESTAMP WHERE id=?",
            (error[:4000], asset_id),
        )
        self.connection.commit()

    def export(self) -> None:
        rows = self.connection.execute(
            """
            SELECT a.*,json_group_array(s.original_path) AS original_paths
            FROM assets a LEFT JOIN sources s ON s.asset_id=a.id
            WHERE a.status='ok' GROUP BY a.id ORDER BY a.id
            """
        )
        json_columns = {
            "metrics_json": "metrics",
            "decision_json": "decision",
            "sampled_frames_json": "sampled_frames",
            "ocr_json": "ocr",
            "understanding_json": "understanding",
            "original_paths": "original_paths",
        }
        with (self.output / "index.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                record = dict(row)
                for old, new in json_columns.items():
                    record[new] = json.loads(record.pop(old) or "null")
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.connection.close()


def discover(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in SUFFIXES)


def process_chunk(
    items: list[WorkItem],
    ocr: PaddleOCRBatch | None,
    vision: QwenVisionBatch | None,
    ocr_batch_size: int,
    vision_batch_size: int,
    min_text_px: float,
) -> dict[int, dict[str, float]]:
    timings = {item.asset_id: {"ocr": 0.0, "vision": 0.0, "encode": 0.0} for item in items}
    for item in items:
        item.metrics = aggregate_metrics([visual_metrics(frame) for frame in item.frames])

    flattened: list[Image.Image] = []
    owners: list[tuple[WorkItem, int]] = []
    for item in items:
        item.ocr_by_frame = []
        for frame_index, frame in zip(item.frame_indices, item.frames):
            flattened.append(frame)
            owners.append((item, frame_index))

    if ocr:
        start = time.perf_counter()
        results = ocr.run(flattened, ocr_batch_size)
        elapsed = time.perf_counter() - start
        per_frame = elapsed / max(1, len(flattened))
        for result, (item, frame_index) in zip(results, owners):
            timings[item.asset_id]["ocr"] += per_frame
            for row in result:
                row["frame_index"] = frame_index
                item.ocr_by_frame.append(row)
    for item in items:
        item.ocr_merged = merge_frame_ocr(item.ocr_by_frame or [])
        item.target_size, item.decision = choose_size(
            item.metrics,
            item.ocr_merged,
            (item.info.width, item.info.height),
            min_text_px,
        )

    if vision:
        representatives = [
            max(item.frames, key=lambda frame: visual_metrics(frame).score)
            for item in items
        ]
        texts = [" ".join(group["text"] for group in (item.ocr_merged or [])) for item in items]
        start = time.perf_counter()
        understandings = vision.run(representatives, texts, vision_batch_size)
        elapsed = time.perf_counter() - start
        for item, understanding in zip(items, understandings):
            item.understanding = understanding
            timings[item.asset_id]["vision"] = elapsed / max(1, len(items))
    else:
        for item in items:
            item.understanding = {}
    return timings


def main() -> None:
    parser = argparse.ArgumentParser(description="表情包压缩、OCR、图片理解和索引流水线")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--device", default="gpu:0")
    parser.add_argument("--sample-frames", type=int, default=7)
    parser.add_argument("--asset-batch-size", type=int, default=8)
    parser.add_argument("--ocr-batch-size", type=int, default=8)
    parser.add_argument("--vision-batch-size", type=int, default=4)
    parser.add_argument("--encode-workers", type=int, default=max(2, min(8, os.cpu_count() or 4)))
    parser.add_argument("--min-text-px", type=float, default=8.0)
    parser.add_argument("--webp-quality", type=int, default=68)
    parser.add_argument("--gif-colors", type=int, default=96)
    parser.add_argument("--no-ocr", action="store_true")
    parser.add_argument("--no-vision", action="store_true")
    parser.add_argument("--vision-model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    input_root = args.input.resolve()
    output_root = args.output.resolve()
    if not input_root.is_dir():
        raise SystemExit(f"输入目录不存在：{input_root}")
    if output_root == input_root or output_root.is_relative_to(input_root):
        raise SystemExit("输出目录必须独立于输入目录")
    output_root.mkdir(parents=True, exist_ok=True)

    store = Store(output_root)
    paths = discover(input_root)
    pending: list[WorkItem] = []
    duplicates = 0
    errors = 0
    for path in paths:
        relative = path.relative_to(input_root).as_posix()
        stat = path.stat()
        digest = file_sha256(path)
        asset_id, complete = store.prepare(relative, digest, stat.st_size, stat.st_mtime_ns)
        if complete and not args.overwrite:
            duplicates += 1
            continue
        try:
            info = inspect_media(path)
            indices, frames = load_keyframes(path, args.sample_frames)
            pending.append(WorkItem(path, relative, digest, asset_id, info, indices, frames))
        except UnidentifiedImageError as exc:
            duplicates += 1
            store.ignore(asset_id, f"{type(exc).__name__}: {exc}")
        except Exception as exc:
            errors += 1
            store.fail(asset_id, f"{type(exc).__name__}: {exc}")

    LOGGER.info("发现 %d，待处理 %d，已完成/重复 %d", len(paths), len(pending), duplicates)
    ocr = None if args.no_ocr else PaddleOCRBatch(args.device)
    vision = None if args.no_vision else QwenVisionBatch(args.vision_model, 640 * 28 * 28)

    processed = 0
    with ThreadPoolExecutor(max_workers=args.encode_workers) as executor:
        for start_index in tqdm(
            range(0, len(pending), args.asset_batch_size),
            unit="批",
            desc="GPU分析 + CPU编码",
        ):
            chunk = pending[start_index : start_index + args.asset_batch_size]
            try:
                timings = process_chunk(
                    chunk,
                    ocr,
                    vision,
                    args.ocr_batch_size,
                    args.vision_batch_size,
                    args.min_text_px,
                )
            except Exception as exc:
                LOGGER.error("批分析失败：%s", exc)
                LOGGER.debug("%s", traceback.format_exc())
                for item in chunk:
                    store.fail(item.asset_id, f"analysis {type(exc).__name__}: {exc}")
                errors += len(chunk)
                continue

            futures = {}
            for item in chunk:
                shard = f"{item.asset_id // 10_000:04d}"
                output_base = output_root / "images" / shard / f"{item.asset_id:012d}"
                dimensions = output_dimensions(item.info.width, item.info.height, item.target_size)
                begin = time.perf_counter()
                future = executor.submit(
                    encode_media,
                    item.path,
                    output_base,
                    item.info,
                    dimensions,
                    args.webp_quality,
                    args.gif_colors,
                )
                futures[future] = (item, begin)
            for future, (item, begin) in futures.items():
                try:
                    output_path = future.result()
                    timings[item.asset_id]["encode"] = time.perf_counter() - begin
                    store.complete(item, output_path, timings[item.asset_id])
                    processed += 1
                except Exception as exc:
                    errors += 1
                    store.fail(item.asset_id, f"encode {type(exc).__name__}: {exc}")
                    LOGGER.error("编码失败 %s：%s", item.relative_path, exc)
    store.export()
    print(
        json.dumps(
            {"found": len(paths), "processed": processed, "skipped_or_duplicate": duplicates, "errors": errors},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
