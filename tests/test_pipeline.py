from types import SimpleNamespace

from universal_emoji_compressor.core import VisualMetrics
from universal_emoji_compressor.pipeline import _reset_to_complexity, text_retention


def test_text_retention_uses_whole_image_character_recall():
    assert text_retention(["轻舟", "已撞大冰山"], ["轻舟已撞", "大冰山"]) == 1.0
    assert text_retention(["我做事你放心"], ["我做事"]) < 0.80
    assert text_retention(["正常文字"], ["完全错误的长文本"]) == 0.0
    assert text_retention(["退下吧！"], ["退下吧", "大", "S"]) == 1.0


def test_no_160_baseline_resets_source_ocr_size_to_complexity():
    item = SimpleNamespace(
        metrics=VisualMetrics(0, 0, 0, 0, 0, 0, 0, 0.30),
        info=SimpleNamespace(width=400, height=400),
        decision={
            "selected": 160,
            "reason": "text",
            "text_constraints": [{"text": "微小水印", "required": 160}],
        },
        target_size=160,
    )

    _reset_to_complexity(item)

    assert item.target_size == 96
    assert item.decision["reason"] == "complexity_no_160_ocr"
    assert item.decision["source_ocr_constraints"][0]["text"] == "微小水印"
