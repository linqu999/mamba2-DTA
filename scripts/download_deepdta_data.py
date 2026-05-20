"""Download the DeepDTA Davis/KIBA benchmark files from GitHub blobs.

The script uses GitHub's blob API instead of raw.githubusercontent.com because
some local networks block or fail DNS for the raw host. Each file is verified
against the Git blob SHA and recorded with a SHA256 checksum.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import time
import json
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.request import urlopen

from _bootstrap import add_src_to_path

add_src_to_path()


REPO_API = "https://api.github.com/repos/hkmztrk/DeepDTA/git/blobs"
SOURCE_REPO = "https://github.com/hkmztrk/DeepDTA"


DATA_FILES = {
    "davis": {
        "Y": {
            "sha": "54c51994801a96c9e808bd897cd93849e6876de0",
            "size": 240605,
        },
        "ligands_can.txt": {
            "sha": "5d252572cc00a6d249edbe5f5175605adc5c6fed",
            "size": 5306,
        },
        "proteins.txt": {
            "sha": "89dea0221c693005aba2f01d18e530542f2a96c7",
            "size": 355211,
        },
        "folds/test_fold_setting1.txt": {
            "sha": "59e42dd65ff3c626f2b323cf64925359196311d4",
            "size": 33252,
        },
        "folds/train_fold_setting1.txt": {
            "sha": "74f49c4891ba1cc703ebd031a36bd4fe839d4c20",
            "size": 166040,
        },
    },
    "kiba": {
        "Y": {
            "sha": "c942af67be84496bf225c119fcc4981c1f4a6915",
            "size": 3867510,
        },
        "ligands_can.txt": {
            "sha": "9a88e2ef7f983cf1cda54d43cfc04de8135f118c",
            "size": 165908,
        },
        "proteins.txt": {
            "sha": "bd19a19f106ea2c30a20dfdbe8db369c82c3d27b",
            "size": 170137,
        },
        "folds/test_fold_setting1.txt": {
            "sha": "d6a015f4443a6caf4edf82102225b4c98b8262a0",
            "size": 139047,
        },
        "folds/train_fold_setting1.txt": {
            "sha": "0c8616f4ccd278b753fb4ccb0d51681d05821543",
            "size": 695885,
        },
    },
}


@dataclass(frozen=True)
class DownloadRecord:
    dataset: str
    relative_path: str
    source_repo: str
    git_blob_sha: str
    expected_size: int
    actual_size: int
    sha256: str
    output_path: str


def fetch_blob(sha: str, retries: int = 4) -> bytes:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with urlopen(f"{REPO_API}/{sha}", timeout=180) as response:
                payload = json.loads(response.read().decode("utf-8"))
            break
        except Exception as exc:  # pragma: no cover - network resilience
            last_error = exc
            if attempt == retries:
                raise
            sleep_seconds = min(2**attempt, 15)
            print(f"retrying blob {sha} after error: {exc!r}; sleep={sleep_seconds}s")
            time.sleep(sleep_seconds)
    else:  # pragma: no cover - defensive
        raise RuntimeError(f"Failed to fetch blob {sha}") from last_error

    if payload["sha"] != sha:
        raise ValueError(f"Blob SHA mismatch: expected {sha}, got {payload['sha']}")
    if payload.get("encoding") != "base64":
        raise ValueError(f"Unexpected blob encoding for {sha}: {payload.get('encoding')}")
    return base64.b64decode(payload["content"])


def write_file(path: Path, content: bytes, expected_size: int) -> str:
    if len(content) != expected_size:
        raise ValueError(f"Size mismatch for {path}: expected {expected_size}, got {len(content)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return hashlib.sha256(content).hexdigest()


def existing_file_sha256(path: Path, expected_size: int) -> str | None:
    if not path.exists() or path.stat().st_size != expected_size:
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download_dataset(dataset: str, output_root: Path) -> list[DownloadRecord]:
    records = []
    for relative_path, metadata in DATA_FILES[dataset].items():
        output_path = output_root / dataset / relative_path
        sha256 = existing_file_sha256(output_path, metadata["size"])
        if sha256 is None:
            content = fetch_blob(metadata["sha"])
            sha256 = write_file(output_path, content, metadata["size"])
            actual_size = len(content)
            print(f"downloaded {dataset}/{relative_path} ({actual_size} bytes)")
        else:
            actual_size = metadata["size"]
            print(f"verified existing {dataset}/{relative_path} ({actual_size} bytes)")
        records.append(
            DownloadRecord(
                dataset=dataset,
                relative_path=relative_path,
                source_repo=SOURCE_REPO,
                git_blob_sha=metadata["sha"],
                expected_size=metadata["size"],
                actual_size=actual_size,
                sha256=sha256,
                output_path=str(output_path),
            )
        )
    return records


def extract_dataset_from_zip(dataset: str, output_root: Path, zip_path: Path) -> list[DownloadRecord]:
    records = []
    with zipfile.ZipFile(zip_path) as archive:
        for relative_path, metadata in DATA_FILES[dataset].items():
            archive_name = f"DeepDTA-master/data/{dataset}/{relative_path}"
            content = archive.read(archive_name)
            output_path = output_root / dataset / relative_path
            sha256 = write_file(output_path, content, metadata["size"])
            records.append(
                DownloadRecord(
                    dataset=dataset,
                    relative_path=relative_path,
                    source_repo=SOURCE_REPO,
                    git_blob_sha=metadata["sha"],
                    expected_size=metadata["size"],
                    actual_size=len(content),
                    sha256=sha256,
                    output_path=str(output_path),
                )
            )
            print(f"extracted {dataset}/{relative_path} ({len(content)} bytes)")
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="data/raw/deepdta")
    parser.add_argument("--manifest", default="data/raw/deepdta/manifest.json")
    parser.add_argument("--dataset", choices=["davis", "kiba", "all"], default="all")
    parser.add_argument("--from-zip", help="Extract files from a DeepDTA repository ZIP instead of downloading blobs.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root)
    datasets = ["davis", "kiba"] if args.dataset == "all" else [args.dataset]

    records: list[DownloadRecord] = []
    for dataset in datasets:
        if args.from_zip:
            records.extend(extract_dataset_from_zip(dataset, output_root, Path(args.from_zip)))
        else:
            records.extend(download_dataset(dataset, output_root))

    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps([asdict(record) for record in records], indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
