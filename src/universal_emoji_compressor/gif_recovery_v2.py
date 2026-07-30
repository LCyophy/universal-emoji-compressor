"""修正版 GIF 恢复安装器：元数据读取本身也允许截断流。"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageFile

from . import core
from . import gif_recovery as base


def inspect_media(path: Path) -> core.MediaInfo:
    previous_setting = ImageFile.LOAD_TRUNCATED_IMAGES
    try:
        # seek 到下一帧时 Pillow 会隐式加载上一帧，因此元数据遍历也需容错。
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        info = base.BASE_INSPECT_MEDIA(path)
    finally:
        ImageFile.LOAD_TRUNCATED_IMAGES = previous_setting

    if info.format != "GIF" or info.frame_count <= 1:
        base.RECOVERY_BY_SOURCE[base._key(path)] = base.RecoveryInfo(
            False, [], info.frame_count, info.frame_count, info.duration_ms, info.duration_ms
        )
        return info

    skipped: list[int] = []
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
            good: list[int] = []
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

    base.RECOVERY_BY_SOURCE[base._key(path)] = base.RecoveryInfo(
        recovered=bool(skipped),
        skipped_frame_indices=skipped,
        original_frame_count=info.frame_count,
        output_frame_count=info.frame_count - len(skipped),
        original_duration_ms=info.duration_ms,
        output_duration_ms=info.duration_ms,
    )
    return info


def install() -> None:
    base.install()
    core.inspect_media = inspect_media
