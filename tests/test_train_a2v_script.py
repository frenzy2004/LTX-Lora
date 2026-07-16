import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def _run(*extra: str) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "train_a2v.py"),
        *extra,
    ]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    environment.pop("FAL_KEY", None)
    environment.pop("LTX_LORA_PRIVATE_ROOT", None)
    return subprocess.run(command, cwd=ROOT, capture_output=True, text=True, env=environment, timeout=30)


def test_a2v_command_exposes_only_safe_immutable_surface() -> None:
    result = _run("--help")

    assert result.returncode == 0, result.stderr
    assert "--run-dir" in result.stdout
    assert "--confirm-bundle-id" in result.stdout
    assert "--execute" in result.stdout
    for legacy in ("--dataset", "--budget", "--steps", "--validation-json", "--approved-plan"):
        assert legacy not in result.stdout


def test_a2v_command_rejects_legacy_dataset_control_flag() -> None:
    result = _run(
        "--run-dir",
        "C:/safe/run",
        "--confirm-bundle-id",
        "0" * 64,
        "--dataset",
        "untrusted.zip",
    )

    assert result.returncode != 0
    assert "unrecognized arguments" in result.stderr
