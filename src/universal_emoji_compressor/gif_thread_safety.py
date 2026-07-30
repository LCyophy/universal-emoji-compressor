"""统一串行化所有会切换 Pillow 全局截断模式的 GIF I/O。"""

from __future__ import annotations

import threading
from pathlib import Path

from . import core
from . import gif_quality_fix as quality

LOCK = threading.RLock()


def install() -> None:
    recovery_inspect = core.inspect_media
    recovery_keyframes = core.load_keyframes

    def inspect_media(path: Path) -> core.MediaInfo:
        with LOCK:
            return recovery_inspect(path)

    def load_keyframes(path: Path, limit: int):
        with LOCK:
            return recovery_keyframes(path, limit)

    def encode_media(
        source: Path,
        output_base: Path,
        info: core.MediaInfo,
        dimensions: tuple[int, int],
        webp_quality: int,
        gif_colors: int,
    ) -> Path:
        if info.format != "GIF":
            return quality.encode_media(
                source, output_base, info, dimensions, webp_quality, gif_colors
            )
        with LOCK:
            return quality.encode_media(
                source, output_base, info, dimensions, webp_quality, gif_colors
            )

    core.inspect_media = inspect_media
    core.load_keyframes = load_keyframes
    core.encode_media = encode_media
