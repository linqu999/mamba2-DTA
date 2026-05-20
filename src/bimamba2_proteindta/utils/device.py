"""Device inspection helpers."""

from __future__ import annotations


def torch_device_summary() -> dict[str, object]:
    try:
        import torch
    except ImportError:
        return {"torch_available": False, "cuda_available": False}

    summary: dict[str, object] = {
        "torch_available": True,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        summary["cuda_version"] = torch.version.cuda
        summary["device_count"] = torch.cuda.device_count()
        summary["device_name"] = torch.cuda.get_device_name(0)
    return summary
