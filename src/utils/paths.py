from __future__ import annotations

from pathlib import Path

import yaml


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_from_root(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else project_root() / path


def build_yolo_data_yaml(config: dict) -> Path:
    output_dir = project_root() / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = output_dir / "dataset_yolo.yaml"

    prepared_root = resolve_from_root(config["data"]["prepared_root"])
    split_names = config["data"]["split_names"]

    payload = {
        "path": str(prepared_root),
        "train": str((prepared_root / "train" / "images").resolve()),
        "val": str((prepared_root / "val" / "images").resolve()),
        "names": config["data"]["class_names"],
    }

    test_dir = prepared_root / "test" / "images"
    if test_dir.exists():
        payload["test"] = str(test_dir.resolve())

    with yaml_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)

    return yaml_path
