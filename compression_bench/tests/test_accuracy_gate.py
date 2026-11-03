"""The correctness gate's unit-test judge: a program that does not reach the end of
`check` passes nothing, whatever its exit status."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness" / "final_validate"))
import accuracy_benchmark as ab  # noqa: E402

ANSWER = {"prompt": "def add(a, b):\n", "entry_point": "add",
          "test": "def check(f):\n    assert f(1, 2) == 3\n    assert f(0, 0) == 0\n"}


def test_a_correct_completion_passes():
    assert ab.passes(ANSWER, "    return a + b\n\n\nif __name__ == '__main__':\n    pass")


def test_a_wrong_completion_fails():
    assert not ab.passes(ANSWER, "    return a - b")


def test_ending_the_process_early_passes_nothing():
    for text in ("    return a - b\nimport os\nos._exit(0)",
                 "    return a - b\nimport sys\nsys.exit(0)",
                 "    import os\n    os._exit(0)",
                 "    raise SystemExit(0)"):
        assert not ab.passes(ANSWER, text), text


def test_a_loop_is_killed_at_the_time_limit(monkeypatch):
    monkeypatch.setattr(ab, "EXEC_TIMEOUT_S", 1)
    assert not ab.passes(ANSWER, "    while True:\n        pass")


def test_the_completion_ends_at_the_first_top_level_statement():
    assert ab.completion_code("    return 1\n\ndef other():\n    pass") == "    return 1\n"
