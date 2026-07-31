from PIL import Image

from universal_emoji_compressor.core import Store, _frame_value
from universal_emoji_compressor.gif_transparency_fix import quantize_rgba
from universal_emoji_compressor.vision import parse_json


def test_frame_value_accepts_scalar_and_per_frame_metadata():
    assert _frame_value(80, 1, 100) == 80
    assert _frame_value([40, 60], 1, 100) == 60
    assert _frame_value([], 0, 100) == 100


def test_opaque_gif_frame_does_not_declare_transparency():
    opaque = quantize_rgba(Image.new("RGBA", (8, 8), (255, 0, 0, 255)), 96)
    transparent = quantize_rgba(Image.new("RGBA", (8, 8), (255, 0, 0, 0)), 96)

    assert "transparency" not in opaque.info
    assert transparent.info["transparency"] == 255


def test_ignored_asset_is_treated_as_complete(tmp_path):
    store = Store(tmp_path)
    asset_id, complete = store.prepare("fake.png", "abc", 4, 1)
    assert not complete

    store.ignore(asset_id, "not an image")
    same_id, complete = store.prepare("fake.png", "abc", 4, 1)
    store.connection.close()

    assert same_id == asset_id
    assert complete


def test_malformed_fenced_model_json_is_salvaged():
    malformed = """```json
{
  "description": "一个卡通角色在吃东西。",
  "emotion": ["开心", "满足"],
  "objects": ["卡通角色", "食物"],
  "tags": ["卡通", "吃东西", "快乐")],
  "safety": "normal"
}
```"""

    result = parse_json(malformed)

    assert result["description"] == "一个卡通角色在吃东西。"
    assert result["tags"] == ["卡通", "吃东西", "快乐"]
    assert result["safety"] == "normal"
