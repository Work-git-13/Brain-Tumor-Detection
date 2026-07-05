from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from .yolo_detection import discover_dataset_layout


def collect_dataset_statistics(dataset_root: str | Path, split_names: dict[str, str]) -> dict:
    layout = discover_dataset_layout(dataset_root, split_names)
    per_split: dict[str, dict] = {}
    overall_boxes = 0
    missing_labels = 0
    empty_labels = 0

    for split_name, samples in layout.split_to_samples.items():
        split_stats = {
            "images": len(samples),
            "boxes": 0,
            "per_class_images": defaultdict(int),
            "per_class_boxes": defaultdict(int),
            "empty_labels": 0,
            "missing_labels": 0,
        }

        for sample in samples:
            split_stats["per_class_images"][sample.class_name] += 1

            if sample.label_path is None:
                split_stats["missing_labels"] += 1
                missing_labels += 1
                continue

            lines = sample.label_path.read_text(encoding="utf-8").strip().splitlines()
            if not lines:
                split_stats["empty_labels"] += 1
                empty_labels += 1
                continue

            box_count = len(lines)
            split_stats["boxes"] += box_count
            split_stats["per_class_boxes"][sample.class_name] += box_count
            overall_boxes += box_count

        split_stats["per_class_images"] = dict(sorted(split_stats["per_class_images"].items()))
        split_stats["per_class_boxes"] = dict(sorted(split_stats["per_class_boxes"].items()))
        per_split[split_name] = split_stats

    return {
        "dataset_root": str(Path(dataset_root).resolve()),
        "class_names": layout.class_names,
        "total_images": layout.total_images,
        "total_boxes": overall_boxes,
        "missing_labels": missing_labels,
        "empty_labels": empty_labels,
        "splits": per_split,
    }

