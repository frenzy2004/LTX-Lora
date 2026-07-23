from ltx_lora_pilot.ltx23_v2 import TRIGGER
from scripts.prepare_ltx23_tutorial_dataset import choose_clip_windows, sanitize_caption


def test_sanitize_caption_removes_blocked_names_and_uses_trigger() -> None:
    caption = sanitize_caption("Realname Surname says hello and talks about coffee.")

    assert caption.startswith(f"{TRIGGER} says, ")
    assert "realname" not in caption.lower()
    assert "surname" not in caption.lower()
    assert "coffee" in caption


def test_sanitize_caption_collapses_noise_and_quotes() -> None:
    caption = sanitize_caption("  This   is   \"quoted\"   speech.  ")

    assert caption == f'{TRIGGER} says, "This is quoted speech."'


def test_choose_clip_windows_prefers_short_sentence_spans() -> None:
    segments = [
        {"start": 0.0, "end": 1.2, "text": "Too short."},
        {"start": 2.0, "end": 6.5, "text": "This is a useful sentence about a notebook."},
        {"start": 8.0, "end": 15.5, "text": "This useful line talks about a sandwich and a meeting."},
        {"start": 20.0, "end": 31.0, "text": "Too long for the tutorial style."},
    ]

    windows = choose_clip_windows(segments, target_count=2, min_seconds=2.0, max_seconds=8.0)

    assert [round(float(window["duration"]), 1) for window in windows] == [4.5, 7.5]
    assert all(str(window["caption"]).startswith(f"{TRIGGER} says, ") for window in windows)


def test_choose_clip_windows_combines_short_neighbors() -> None:
    segments = [
        {"start": 1.0, "end": 2.0, "text": "First short line,"},
        {"start": 2.1, "end": 4.2, "text": "then it becomes useful."},
        {"start": 5.0, "end": 7.3, "text": "Another complete line."},
    ]

    windows = choose_clip_windows(segments, target_count=2, min_seconds=2.0, max_seconds=8.0)

    assert windows[0]["start"] == 1.0
    assert windows[0]["end"] == 4.2
    assert "First short line, then it becomes useful." in str(windows[0]["caption"])
