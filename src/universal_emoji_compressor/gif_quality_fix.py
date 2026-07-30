"""避免 Pillow optimize=True 对局部调色板 GIF 造成索引重映射花屏。"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageFile

from . import core
from . import gif_recovery as recovery

BASE_ENCODE_MEDIA = core.encode_media


def encode_media(
    source: Path,
    output_base: Path,
    info: core.MediaInfo,
    dimensions: tuple[int, int],
    webp_quality: int,
    gif_colors: int,
) -> Path:
    if info.format != "GIF" or not info.animated:
        return BASE_ENCODE_MEDIA(
            source, output_base, info, dimensions, webp_quality, gif_colors
        )

    recovery_info = recovery.RECOVERY_BY_SOURCE.get(recovery._key(source))
    skipped = (
        set(recovery_info.skipped_frame_indices)
        if recovery_info and recovery_info.recovered
        else set()
    )
    good_indices, durations = recovery._redistributed_durations(info, skipped)
    output = output_base.with_suffix(".gif")
    output.parent.mkdir(parents=True, exist_ok=True)
    frames: list[Image.Image] = []
    disposals: list[int] = []
    previous_setting = ImageFile.LOAD_TRUNCATED_IMAGES
    try:
        ImageFile.LOAD_TRUNCATED_IMAGES = bool(skipped)
        with Image.open(source) as image:
            for index in good_indices:
                image.seek(index)
                frame = image.convert("RGBA")
                frame.load()
                if frame.size != dimensions:
                    frame = frame.resize(dimensions, Image.Resampling.LANCZOS)
                frames.append(
                    frame.convert(
                        "P",
                        palette=Image.Palette.ADAPTIVE,
                        colors=gif_colors,
                    )
                )
                disposals.append(info.disposals[index])
    finally:
        ImageFile.LOAD_TRUNCATED_IMAGES = previous_setting

    if len(frames) != len(good_indices):
        raise OSError("GIF 有效帧读取数量不一致")
    frames[0].save(
        output,
        "GIF",
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=info.loop,
        disposal=disposals,
        optimize=False,
    )
    encoded = recovery.BASE_INSPECT_MEDIA(output)
    if encoded.duration_ms != info.duration_ms:
        raise OSError(
            f"GIF 输出总时长不一致：{encoded.duration_ms} != {info.duration_ms}"
        )
    if recovery_info:
        recovery_info.output_frame_count = encoded.frame_count
        recovery_info.output_duration_ms = encoded.duration_ms
    return output


def install() -> None:
    core.encode_media = encode_media
