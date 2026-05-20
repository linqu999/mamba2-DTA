"""Print local environment status for BiMamba2-ProteinDTA."""

from __future__ import annotations

import importlib.util
import platform
import sys
from pathlib import Path


def module_status(module_name: str) -> dict[str, object]:
    spec = importlib.util.find_spec(module_name)
    status: dict[str, object] = {"available": spec is not None}
    if spec is None:
        return status

    try:
        module = __import__(module_name)
    except Exception as exc:  # pragma: no cover - defensive environment probe
        status["import_error"] = repr(exc)
        return status

    version = getattr(module, "__version__", None)
    if version is not None:
        status["version"] = version
    return status


def print_status(name: str, status: dict[str, object]) -> None:
    available = "yes" if status.get("available") else "no"
    version = status.get("version", "")
    suffix = f" ({version})" if version else ""
    print(f"{name}: {available}{suffix}")
    if "import_error" in status:
        print(f"  import_error: {status['import_error']}")


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    print("BiMamba2-ProteinDTA environment check")
    print(f"repo_root: {repo_root}")
    print(f"python: {sys.version.split()[0]}")
    print(f"platform: {platform.platform()}")

    for module_name in ["numpy", "pandas", "yaml", "torch", "mamba_ssm"]:
        print_status(module_name, module_status(module_name))

    torch_status = module_status("torch")
    if torch_status.get("available") and "import_error" not in torch_status:
        import torch

        print(f"cuda_available: {torch.cuda.is_available()}")
        print(f"cuda_version: {torch.version.cuda}")
        print(f"cuda_device_count: {torch.cuda.device_count()}")
        if torch.cuda.is_available():
            print(f"cuda_device_0: {torch.cuda.get_device_name(0)}")
    else:
        print("cuda_available: no (torch unavailable)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
