"""为局部调色板保留统一透明索引 255。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageFile

from . import core
from . import gif_recovery as recovery
from .gif_thread_safety import LOCK

BASE_ENCODE_MEDIA = core.encode_media


def quantize_rgba(frame: Image.Image, colors: int) -> Image.Image:
    rgba = frame.convert("RGBA")
    alpha = np.asarray(rgba.getchannel("A"), dtype=np.uint8)
    quantized = rgba.convert("RGB").quantize(
        colors=max(2, min(255, colors - 1)),
        method=Image.Quantize.MEDIANCUT,
        dither=Image.Dither.FLOYDSTEINBERG,
    )
    palette = list(quantized.getpalette() or [])
    palette.extend([0] * (768 - len(palette)))
    palette[255 * 3 : 255 * 3 + 3] = [0, 0, 0]
    indices = np.asarray(quantized, dtype=np.uint8).copy()
    transparent = alpha < 128
    indices[transparent] = 255
    output = Image.fromarray(indices, mode="P")
    output.putpalette(palette)
    if bool(transparent.any()):
        output.info["transparency"] = 255
    return output


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
    with LOCK:
        recovery_info = recovery.RECOVERY_BY_SOURCE.get(recovery._key(source))
        skipped = (
            set(recovery_info.skipped_frame_indices)
            if recovery_info and recovery_info.recovered
            else set()
        )
        good_indices, durations = recovery._redistributed_durations(info, skipped)
        timed_frames = [
            (index, duration)
            for index, duration in zip(good_indices, durations, strict=True)
            if duration > 0
        ]
        if timed_frames:
            good_indices = [index for index, _ in timed_frames]
            durations = [duration for _, duration in timed_frames]
        else:
            good_indices = good_indices[:1]
            durations = [0]
        frames: list[Image.Image] = []
        disposals: list[int] = []
        previous = ImageFile.LOAD_TRUNCATED_IMAGES
        try:
            ImageFile.LOAD_TRUNCATED_IMAGES = bool(skipped)
            with Image.open(source) as image:
                for index in good_indices:
                    image.seek(index)
                    frame = image.convert("RGBA")
                    frame.load()
                    if frame.size != dimensions:
                        frame = frame.resize(dimensions, Image.Resampling.LANCZOS)
                    frames.append(quantize_rgba(frame, gif_colors))
                    disposals.append(info.disposals[index])
        finally:
            ImageFile.LOAD_TRUNCATED_IMAGES = previous

        output = output_base.with_suffix(".gif")
        output.parent.mkdir(parents=True, exist_ok=True)
        collapsed_to_single = len(frames) > 1 and all(
            frame.copy().convert("RGBA").tobytes()
            == frames[0].copy().convert("RGBA").tobytes()
            for frame in frames[1:]
        )
        single_frame_output = len(frames) == 1 or collapsed_to_single
        append_images = [] if single_frame_output else frames[1:]
        save_duration = sum(durations) if single_frame_output else durations
        save_disposal = disposals[0] if single_frame_output else disposals
        frames[0].save(
            output,
            "GIF",
            save_all=True,
            append_images=append_images,
            duration=save_duration,
            loop=info.loop,
            disposal=save_disposal,
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
