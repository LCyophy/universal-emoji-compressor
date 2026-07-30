"""GIF 坏帧恢复：跳帧但保持总时长，并将恢复信息写入索引。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import numpy as np
from PIL import Image, ImageFile

from . import core

BASE_INSPECT_MEDIA = core.inspect_media
BASE_LOAD_KEYFRAMES = core.load_keyframes
BASE_ENCODE_MEDIA = core.encode_media
BASE_STORE_INIT = core.Store.__init__
BASE_STORE_COMPLETE = core.Store.complete


@dataclass(slots=True)
class RecoveryInfo:
    recovered: bool
    skipped_frame_indices: list[int]
    original_frame_count: int
    output_frame_count: int
    original_duration_ms: int
    output_duration_ms: int


RECOVERY_BY_SOURCE: dict[str, RecoveryInfo] = {}


def _key(path: Path) -> str:
    return str(path.resolve()).casefold()


def inspect_media(path: Path) -> core.MediaInfo:
    info = BASE_INSPECT_MEDIA(path)
    if info.format != "GIF" or info.frame_count <= 1:
        RECOVERY_BY_SOURCE[_key(path)] = RecoveryInfo(
            False, [], info.frame_count, info.frame_count, info.duration_ms, info.duration_ms
        )
        return info

    skipped: list[int] = []
    previous_setting = ImageFile.LOAD_TRUNCATED_IMAGES
    try:
        ImageFile.LOAD_TRUNCATED_IMAGES = False
        with Image.open(path) as image:
            for index in range(info.frame_count):
                try:
                    image.seek(index)
                    frame = image.convert("RGBA")
                    frame.load()
                except (OSError, ValueError):
                    skipped.append(index)

        if skipped:
            ImageFile.LOAD_TRUNCATED_IMAGES = True
            good = []
            with Image.open(path) as image:
                for index in range(info.frame_count):
                    if index in skipped:
                        continue
                    try:
                        image.seek(index)
                        frame = image.convert("RGBA")
                        frame.load()
                        good.append(index)
                    except (OSError, ValueError):
                        skipped.append(index)
            skipped = sorted(set(skipped))
            if not good:
                raise OSError("GIF 所有帧均无法恢复")
    finally:
        ImageFile.LOAD_TRUNCATED_IMAGES = previous_setting

    recovered = bool(skipped)
    RECOVERY_BY_SOURCE[_key(path)] = RecoveryInfo(
        recovered=recovered,
        skipped_frame_indices=skipped,
        original_frame_count=info.frame_count,
        output_frame_count=info.frame_count - len(skipped),
        original_duration_ms=info.duration_ms,
        output_duration_ms=info.duration_ms,
    )
    return info


def _select_good_keyframes(
    path: Path,
    frame_count: int,
    skipped: set[int],
    limit: int,
) -> tuple[list[int], list[Image.Image]]:
    good_indices = [index for index in range(frame_count) if index not in skipped]
    if not good_indices:
        raise OSError("GIF 没有可采样的有效帧")
    previous_setting = ImageFile.LOAD_TRUNCATED_IMAGES
    thumbnails: dict[int, np.ndarray] = {}
    try:
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        with Image.open(path) as image:
            for index in good_indices:
                image.seek(index)
                thumb = image.convert("L")
                thumb.thumbnail((64, 64), Image.Resampling.BILINEAR)
                canvas = Image.new("L", (64, 64))
                canvas.paste(thumb, ((64 - thumb.width) // 2, (64 - thumb.height) // 2))
                thumbnails[index] = np.asarray(canvas, dtype=np.float32)

        uniform_positions = core._uniform_indices(len(good_indices), max(2, round(limit * 0.65)))
        selected = {good_indices[position] for position in uniform_positions}
        differences = []
        for previous, current in pairwise(good_indices):
            difference = float(np.mean(np.abs(thumbnails[current] - thumbnails[previous])))
            differences.append((difference, current))
        for _, index in sorted(differences, reverse=True):
            if len(selected) >= min(limit, len(good_indices)):
                break
            selected.add(index)

        selected_indices = sorted(selected)
        frames: list[Image.Image] = []
        with Image.open(path) as image:
            for index in selected_indices:
                image.seek(index)
                frames.append(image.convert("RGBA").copy())
        return selected_indices, frames
    finally:
        ImageFile.LOAD_TRUNCATED_IMAGES = previous_setting


def load_keyframes(path: Path, limit: int) -> tuple[list[int], list[Image.Image]]:
    recovery = RECOVERY_BY_SOURCE.get(_key(path))
    if not recovery or not recovery.recovered:
        return BASE_LOAD_KEYFRAMES(path, limit)
    return _select_good_keyframes(
        path,
        recovery.original_frame_count,
        set(recovery.skipped_frame_indices),
        limit,
    )


def _redistributed_durations(info: core.MediaInfo, skipped: set[int]) -> tuple[list[int], list[int]]:
    good = [index for index in range(info.frame_count) if index not in skipped]
    if not good:
        raise OSError("GIF 所有帧均损坏")
    durations = {index: info.durations[index] for index in good}
    for bad_index in sorted(skipped):
        previous = next((index for index in reversed(good) if index < bad_index), None)
        recipient = previous
        if recipient is None:
            recipient = next((index for index in good if index > bad_index), None)
        if recipient is None:
            raise OSError(f"坏帧 {bad_index} 无法转移时长")
        durations[recipient] += info.durations[bad_index]
    return good, [durations[index] for index in good]


def encode_media(
    source: Path,
    output_base: Path,
    info: core.MediaInfo,
    dimensions: tuple[int, int],
    webp_quality: int,
    gif_colors: int,
) -> Path:
    recovery = RECOVERY_BY_SOURCE.get(_key(source))
    if info.format != "GIF" or not recovery or not recovery.recovered:
        return BASE_ENCODE_MEDIA(
            source, output_base, info, dimensions, webp_quality, gif_colors
        )

    skipped = set(recovery.skipped_frame_indices)
    good_indices, durations = _redistributed_durations(info, skipped)
    output = output_base.with_suffix(".gif")
    output.parent.mkdir(parents=True, exist_ok=True)
    frames: list[Image.Image] = []
    disposals: list[int] = []
    previous_setting = ImageFile.LOAD_TRUNCATED_IMAGES
    try:
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        with Image.open(source) as image:
            for index in good_indices:
                image.seek(index)
                frame = image.convert("RGBA")
                frame.load()
                if frame.size != dimensions:
                    frame = frame.resize(dimensions, Image.Resampling.LANCZOS)
                frames.append(
                    frame.convert("P", palette=Image.Palette.ADAPTIVE, colors=gif_colors)
                )
                disposals.append(info.disposals[index])
    finally:
        ImageFile.LOAD_TRUNCATED_IMAGES = previous_setting
    if len(frames) != len(good_indices):
        raise OSError("跳过坏帧后无法继续合成 GIF")
    frames[0].save(
        output,
        "GIF",
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=info.loop,
        disposal=disposals,
        optimize=True,
    )

    encoded = BASE_INSPECT_MEDIA(output)
    if encoded.duration_ms != info.duration_ms:
        raise OSError(
            f"GIF 恢复后总时长不一致：{encoded.duration_ms} != {info.duration_ms}"
        )
    recovery.output_frame_count = encoded.frame_count
    recovery.output_duration_ms = encoded.duration_ms
    return output


def store_init(self: core.Store, output: Path) -> None:
    BASE_STORE_INIT(self, output)
    existing = {
        row[1] for row in self.connection.execute("PRAGMA table_info(assets)")
    }
    additions = {
        "recovered": "INTEGER NOT NULL DEFAULT 0",
        "skipped_frame_indices_json": "TEXT",
        "original_frame_count": "INTEGER",
        "output_frame_count": "INTEGER",
        "original_duration_ms": "INTEGER",
        "output_duration_ms": "INTEGER",
    }
    for column, declaration in additions.items():
        if column not in existing:
            self.connection.execute(
                f"ALTER TABLE assets ADD COLUMN {column} {declaration}"
            )
    self.connection.commit()


def store_complete(
    self: core.Store,
    item: core.WorkItem,
    output_path: Path,
    timings: dict[str, float],
) -> None:
    BASE_STORE_COMPLETE(self, item, output_path, timings)
    recovery = RECOVERY_BY_SOURCE.get(_key(item.path))
    if recovery is None:
        recovery = RecoveryInfo(
            False,
            [],
            item.info.frame_count,
            item.info.frame_count,
            item.info.duration_ms,
            item.info.duration_ms,
        )
    self.connection.execute(
        """
        UPDATE assets SET recovered=?,skipped_frame_indices_json=?,
         original_frame_count=?,output_frame_count=?,
         original_duration_ms=?,output_duration_ms=?
        WHERE id=?
        """,
        (
            int(recovery.recovered),
            json.dumps(recovery.skipped_frame_indices),
            recovery.original_frame_count,
            recovery.output_frame_count,
            recovery.original_duration_ms,
            recovery.output_duration_ms,
            item.asset_id,
        ),
    )
    self.connection.commit()


def store_export(self: core.Store) -> None:
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
        "skipped_frame_indices_json": "skipped_frame_indices",
        "original_paths": "original_paths",
    }
    with (self.output / "index.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            record = dict(row)
            for old, new in json_columns.items():
                record[new] = json.loads(record.pop(old) or "null")
            record["recovered"] = bool(record["recovered"])
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    self.connection.close()


def install() -> None:
    core.inspect_media = inspect_media
    core.load_keyframes = load_keyframes
    core.encode_media = encode_media
    core.Store.__init__ = store_init
    core.Store.complete = store_complete
    core.Store.export = store_export
