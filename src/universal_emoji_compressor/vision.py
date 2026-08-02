"""第二阶段：独立 PyTorch 进程批量生成图片理解索引并回填 SQLite。"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import time
from pathlib import Path

from PIL import Image, ImageDraw
from tqdm import tqdm


def representative_image(path: Path, tile_size: int = 256) -> Image.Image:
    """Build a visual contact sheet while tolerating damaged animation frames.

    A GIF may be previewable even when one or more compressed frames raise
    ``broken data stream``. Decode sampled frames independently and omit only
    the bad samples; scan all frames as a fallback when the selected samples
    are all damaged. The compression stage remains responsible for preserving
    the original animation duration and recording skipped-frame metadata.
    """
    with Image.open(path) as image:
        count = int(getattr(image, "n_frames", 1))
        if count <= 1:
            try:
                image.seek(0)
                image.load()
                return image.convert("RGB").copy()
            except (OSError, ValueError):
                raise OSError(f"no decodable frame in {path}")

        sample_count = min(4, count)
        indices = sorted(
            {
                round(index * (count - 1) / (sample_count - 1))
                for index in range(sample_count)
            }
        )
        tiles: list[tuple[int, Image.Image]] = []
        for index in indices:
            try:
                image.seek(index)
                image.load()
                frame = image.convert("RGB")
            except (OSError, ValueError):
                continue
            frame.thumbnail((tile_size, tile_size), Image.Resampling.LANCZOS)
            tile = Image.new("RGB", (tile_size, tile_size), "white")
            tile.paste(frame, ((tile_size - frame.width) // 2, (tile_size - frame.height) // 2))
            ImageDraw.Draw(tile).text((5, 5), f"#{index}", fill="red")
            tiles.append((index, tile))

        if not tiles:
            for index in range(count):
                if index in indices:
                    continue
                try:
                    image.seek(index)
                    image.load()
                    frame = image.convert("RGB")
                except (OSError, ValueError):
                    continue
                frame.thumbnail((tile_size, tile_size), Image.Resampling.LANCZOS)
                tile = Image.new("RGB", (tile_size, tile_size), "white")
                tile.paste(frame, ((tile_size - frame.width) // 2, (tile_size - frame.height) // 2))
                ImageDraw.Draw(tile).text((5, 5), f"#{index}", fill="red")
                tiles.append((index, tile))
                if len(tiles) >= sample_count:
                    break

    if not tiles:
        raise OSError(f"no decodable frame in {path}")
    columns = 2
    rows = (len(tiles) + 1) // 2
    sheet = Image.new("RGB", (tile_size * columns, tile_size * rows), "white")
    for index, (_, tile) in enumerate(tiles):
        sheet.paste(tile, ((index % columns) * tile_size, (index // columns) * tile_size))
    return sheet


def _decode_json_string(value: str) -> str:
    try:
        return str(json.loads(f'"{value}"'))
    except json.JSONDecodeError:
        return value.replace('\\"', '"').replace('\\n', ' ').strip()


def _extract_string(text: str, key: str) -> str:
    match = re.search(
        rf'"{re.escape(key)}"\s*:\s*"((?:\\.|[^"\\])*)"',
        text,
        re.DOTALL,
    )
    return _decode_json_string(match.group(1)).strip() if match else ""


def _extract_array(text: str, *keys: str) -> list[str]:
    boundary = r'(?=\n\s*"(?:description|emotion|objects|tags|标签|safety)"\s*:|\Z)'
    for key in keys:
        match = re.search(
            rf'"{re.escape(key)}"\s*\]?\s*[:=]\s*\[(.*?)(?:\]|{boundary})',
            text,
            re.DOTALL,
        )
        if not match:
            continue
        values = [
            _decode_json_string(value).strip()
            for value in re.findall(r'"((?:\\.|[^"\\])*)"', match.group(1))
        ]
        return list(dict.fromkeys(value for value in values if value))[:10]
    return []


def parse_json(text: str) -> dict:
    result: dict
    try:
        begin, end = text.index("{"), text.rindex("}") + 1
        decoded = json.loads(text[begin:end])
        result = decoded if isinstance(decoded, dict) else {}
    except (ValueError, json.JSONDecodeError):
        result = {
            "description": _extract_string(text, "description"),
            "emotion": _extract_array(text, "emotion"),
            "objects": _extract_array(text, "objects"),
            "tags": _extract_array(text, "tags", "标签"),
            "safety": _extract_string(text, "safety") or "unknown",
        }
    for key in ("emotion", "objects", "tags"):
        value = result.get(key)
        if not isinstance(value, list):
            value = [str(value)] if value else []
        result[key] = list(dict.fromkeys(str(item).strip() for item in value if str(item).strip()))[:10]
    result["description"] = str(result.get("description", "")).strip()
    result["safety"] = str(result.get("safety", "unknown")).lower()
    if not result["tags"]:
        result["tags"] = (result["objects"] + result["emotion"] + ["表情包"])[:10]
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_root", type=Path)
    parser.add_argument("model", help="本地模型目录或 Hugging Face 模型名")
    parser.add_argument("--source-root", type=Path, help="优先用原图做视觉理解")
    parser.add_argument("--context", default="", help="可选的数据集主题背景，不作为识别结果直接复制")
    parser.add_argument("--device", default="auto", help="auto、cuda:0 或 cpu")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-pixels", type=int, default=320 * 32 * 32)
    parser.add_argument("--asset-id", type=int, action="append", help="只处理指定资产 ID，可重复")
    parser.add_argument("--limit", type=int, help="最多处理多少条，用于小样验证")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    connection = sqlite3.connect(args.output_root / "index.sqlite")
    connection.row_factory = sqlite3.Row
    condition = "status='ok'"
    if not args.overwrite:
        condition += " AND (understanding_json IS NULL OR understanding_json='' OR understanding_json='{}')"
    assets = list(
        connection.execute(
            f"""SELECT a.id,a.output_path,a.ocr_text,
             (SELECT s.original_path FROM sources s WHERE s.asset_id=a.id ORDER BY s.id LIMIT 1)
             AS original_path FROM assets a WHERE {condition} ORDER BY a.id"""
        )
    )
    if args.asset_id:
        selected_ids = set(args.asset_id)
        assets = [row for row in assets if row["id"] in selected_ids]
    if args.limit is not None:
        assets = assets[: max(0, args.limit)]

    load_started = time.perf_counter()
    device = "cuda:0" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        dtype=dtype,
        device_map=device if device.startswith("cuda") else None,
    )
    if not device.startswith("cuda"):
        model.to(device)
    processor = AutoProcessor.from_pretrained(
        args.model,
        min_pixels=96 * 32 * 32,
        max_pixels=args.max_pixels,
        use_fast=True,
    )
    if hasattr(processor, "tokenizer"):
        processor.tokenizer.padding_side = "left"
    load_seconds = time.perf_counter() - load_started
    prompt = (
        "分析这个表情包（若是四格图，四格是同一个 GIF 的不同时间帧）。"
        "只输出合法 JSON，不要 Markdown。字段："
        "description（一句具体的简体中文描述）；"
        "emotion（情绪数组）；objects（人物、动物或物体数组）；"
        "tags（5到10个适合搜索的简体中文标签）；"
        "safety 字段必须存在，且只能是 normal、adult 或 violence。"
        + (f"数据集背景（只能辅助判断，仍以画面为准）：{args.context}。" if args.context else "")
        + "已识别文字："
    )

    inference_seconds = 0.0
    results: list[dict] = []
    for start in tqdm(range(0, len(assets), args.batch_size), unit="批", desc="Qwen-VL"):
        batch = assets[start : start + args.batch_size]
        image_paths = []
        for row in batch:
            source_path = (
                args.source_root / Path(row["original_path"])
                if args.source_root and row["original_path"]
                else None
            )
            image_paths.append(
                source_path
                if source_path is not None and source_path.is_file()
                else args.output_root / Path(row["output_path"])
            )
        images = [representative_image(path) for path in image_paths]
        conversations = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt + (row["ocr_text"] or "无")},
                    ],
                }
            ]
            for row, image in zip(batch, images)
        ]
        texts = [
            processor.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True,
            )
            for conversation in conversations
        ]
        inputs = processor(
            text=texts,
            images=images,
            padding=True,
            return_tensors="pt",
        ).to(model.device)
        started = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=180,
                do_sample=False,
            )
        batch_seconds = time.perf_counter() - started
        inference_seconds += batch_seconds
        trimmed = [
            output[len(source) :]
            for source, output in zip(inputs.input_ids, generated)
        ]
        decoded = processor.batch_decode(trimmed, skip_special_tokens=True)
        for row, text in zip(batch, decoded):
            understanding = parse_json(text)
            seconds = batch_seconds / max(1, len(batch))
            connection.execute(
                """
                UPDATE assets SET understanding_json=?,vision_seconds=?,
                 updated_at=CURRENT_TIMESTAMP WHERE id=?
                """,
                (json.dumps(understanding, ensure_ascii=False), seconds, row["id"]),
            )
            tags = []
            for key in ("emotion", "objects", "tags"):
                tags.extend(map(str, understanding.get(key, [])))
            connection.execute("DELETE FROM asset_search WHERE asset_id=?", (row["id"],))
            connection.execute(
                "INSERT INTO asset_search(asset_id,ocr_text,description,tags) VALUES(?,?,?,?)",
                (row["id"], row["ocr_text"] or "", understanding["description"], " ".join(tags)),
            )
            results.append({"asset_id": row["id"], **understanding, "seconds": round(seconds, 4)})
        connection.commit()

    valid_json = sum(bool(row["description"]) and bool(row["tags"]) for row in results)
    report = {
        "model": str(args.model),
        "device": torch.cuda.get_device_name(0) if device.startswith("cuda") else device,
        "model_load_seconds": round(load_seconds, 3),
        "assets": len(results),
        "batch_size": args.batch_size,
        "inference_seconds": round(inference_seconds, 3),
        "images_per_second": round(len(results) / max(inference_seconds, 1e-9), 4),
        "complete_records": valid_json,
        "complete_rate": round(valid_json / max(1, len(results)), 4),
        "results": results,
    }
    report_path = args.report or args.output_root / "vision_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    connection.close()
    print(json.dumps({key: value for key, value in report.items() if key != "results"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
