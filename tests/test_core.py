from universal_emoji_compressor.core import (
    MediaInfo,
    VisualMetrics,
    choose_size,
    normalize_text,
    output_dimensions,
)
from universal_emoji_compressor.gif_recovery import _redistributed_durations


def test_output_dimensions_never_upscales():
    assert output_dimensions(40, 20, 160) == (40, 20)
    assert output_dimensions(400, 200, 160) == (160, 80)


def test_text_constraint_promotes_size():
    metrics = VisualMetrics(0.1, 0.1, 0.1, 0.1, 0.1, 0, 0.1, 0.1)
    groups = [{
        "text": "小字",
        "normalized": "小字",
        "best_score": 0.99,
        "box": [[0, 0], [20, 0], [20, 10], [0, 10]],
    }]
    selected, decision = choose_size(metrics, groups, (400, 400), 10)
    assert selected == 160
    assert decision["reason"] == "text"


def test_normalize_chinese_search_text():
    assert normalize_text(" 你好，世界！ ") == "你好世界"


def test_bad_frame_durations_are_redistributed():
    info = MediaInfo("GIF", 100, 100, 5, [100, 120, 80, 60, 40], 0, [0] * 5)
    indices, durations = _redistributed_durations(info, {0, 2, 3})
    assert indices == [1, 4]
    assert durations == [360, 40]
    assert sum(durations) == info.duration_ms
