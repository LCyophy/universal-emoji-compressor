from pathlib import Path

from PIL import Image

from universal_emoji_compressor import cli, core


def gif_meta(path: Path) -> tuple[int, int, int]:
    with Image.open(path) as image:
        frame_count = int(getattr(image, "n_frames", 1))
        total = 0
        for index in range(frame_count):
            image.seek(index)
            total += int(image.info.get("duration", 100))
        return frame_count, total, int(image.info.get("loop", 0))


def test_gif_keeps_total_duration_and_loop(tmp_path):
    source = tmp_path / "source.gif"
    output_base = tmp_path / "images" / "000000000001"
    frames = [Image.new("RGBA", (96, 64), color) for color in ("red", "green", "blue")]
    frames[0].save(
        source,
        "GIF",
        save_all=True,
        append_images=frames[1:],
        duration=[80, 120, 160],
        loop=3,
        disposal=[2, 2, 2],
    )

    cli.install_runtime()
    info = core.inspect_media(source)
    output = core.encode_media(source, output_base, info, (64, 43), 68, 96)

    _, total, loop = gif_meta(output)
    assert total == 360
    assert loop == 3


def test_single_frame_gif_becomes_webp(tmp_path):
    source = tmp_path / "still.gif"
    Image.new("RGB", (80, 80), "yellow").save(source, "GIF")
    output_base = tmp_path / "images" / "000000000002"

    cli.install_runtime()
    info = core.inspect_media(source)
    output = core.encode_media(source, output_base, info, (64, 64), 68, 96)

    assert output.suffix == ".webp"
    assert output.is_file()
