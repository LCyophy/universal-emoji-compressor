"""跨平台完整流程调度器：压缩/OCR、视觉索引、JSONL 导出。"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run_checked(arguments: list[str]) -> None:
    print("+", subprocess.list2cmdline(arguments), flush=True)
    completed = subprocess.run(arguments, check=False)
    if completed.returncode:
        raise SystemExit(completed.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description="运行完整的表情包压缩与索引流程")
    parser.add_argument("input", type=Path, help="递归扫描的输入目录")
    parser.add_argument("output", type=Path, help="独立输出目录")
    parser.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct", help="本地模型目录或 Hugging Face 模型名")
    parser.add_argument("--device", default="gpu:0", help="PaddleOCR 设备，例如 gpu:0 或 cpu")
    parser.add_argument("--ocr-batch-size", type=int, default=8)
    parser.add_argument("--vision-batch-size", type=int, default=2)
    parser.add_argument("--context", default="", help="可选的数据集主题背景")
    parser.add_argument("--sample-frames", type=int, default=7)
    parser.add_argument("--min-text-px", type=float, default=10.0)
    parser.add_argument("--skip-vision", action="store_true", help="只压缩和建立 OCR 索引")
    parser.add_argument("--no-ocr", action="store_true", help="禁用 OCR，尺寸仅按视觉复杂度选择")
    args = parser.parse_args()

    compress = [
        sys.executable, "-m", "universal_emoji_compressor.cli",
        str(args.input), str(args.output),
        "--device", args.device,
        "--ocr-batch-size", str(args.ocr_batch_size),
        "--asset-batch-size", "8",
        "--sample-frames", str(args.sample_frames),
        "--min-text-px", str(args.min_text_px),
    ]
    if args.no_ocr:
        compress.append("--no-ocr")
    run_checked(compress)

    if not args.skip_vision:
        vision = [
            sys.executable, "-m", "universal_emoji_compressor.vision",
            str(args.output), args.model,
            "--batch-size", str(args.vision_batch_size),
            "--source-root", str(args.input),
        ]
        if args.context:
            vision.extend(["--context", args.context])
        run_checked(vision)
    run_checked([sys.executable, "-m", "universal_emoji_compressor.export", str(args.output)])


if __name__ == "__main__":
    main()
