"""压缩、OCR 与基础索引入口。"""

from __future__ import annotations

import sys

from . import core
from . import pipeline as _pipeline  # noqa: F401  # 安装 OCR 缩放后可读性闭环
from .gif_quality_fix import install as install_gif_quality
from .gif_recovery_v2 import install as install_recovery
from .gif_thread_safety import install as install_gif_thread_safety
from .gif_transparency_fix import install as install_gif_transparency
from .paddle_runtime import install as install_paddle_runtime


def install_runtime() -> None:
    """按经过回归验证的顺序安装 GIF、OCR 运行时补丁。"""
    install_recovery()
    install_gif_quality()
    install_gif_thread_safety()
    install_gif_transparency()
    install_paddle_runtime()


def main() -> None:
    install_runtime()
    # 图片理解单独运行，避免 Paddle 与 PyTorch 在 Windows 争用 cuDNN DLL。
    if "--no-vision" not in sys.argv:
        sys.argv.append("--no-vision")
    core.main()


if __name__ == "__main__":
    main()
