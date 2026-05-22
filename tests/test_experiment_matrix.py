import subprocess
import sys

from scripts.run_experiment_matrix import build_command, load_configs


def test_load_experiment_matrix_ignores_comments_and_blanks(tmp_path) -> None:
    matrix = tmp_path / "matrix.txt"
    matrix.write_text(
        "\n"
        "# comment\n"
        "configs/a.yaml\n"
        "  configs/b.yaml  \n",
        encoding="utf-8",
    )

    assert load_configs(matrix) == ["configs/a.yaml", "configs/b.yaml"]


def test_build_matrix_command_applies_overrides() -> None:
    class Args:
        epochs = 1
        limit_batches = 2
        device = "cuda"

    command = build_command("configs/a.yaml", Args())

    assert command == [
        sys.executable,
        "scripts/03_train.py",
        "--config",
        "configs/a.yaml",
        "--epochs",
        "1",
        "--limit-batches",
        "2",
        "--device",
        "cuda",
    ]


def test_matrix_script_help_runs() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/run_experiment_matrix.py", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--matrix" in completed.stdout
