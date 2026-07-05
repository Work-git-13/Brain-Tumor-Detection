from __future__ import annotations

import shutil
from pathlib import Path

from .yolo_detection import discover_dataset_layout


def prepare_flat_yolo_dataset(config: dict) -> dict:
    dataset_root = Path(config["data"]["dataset_root"])
    prepared_root = Path(config["data"]["prepared_root"])
    split_names = config["data"]["split_names"]
    layout = discover_dataset_layout(dataset_root, split_names)

    prepared_root.mkdir(parents=True, exist_ok=True)
    copied_images = 0
    copied_labels = 0

    for split_name, samples in layout.split_to_samples.items():
        images_out = prepared_root / split_name / "images"
        labels_out = prepared_root / split_name / "labels"
        images_out.mkdir(parents=True, exist_ok=True)
        labels_out.mkdir(parents=True, exist_ok=True)

        for sample in samples:
            target_stem = f"{sample.class_name.lower().replace(' ', '_')}__{sample.image_path.stem}"
            image_target = images_out / f"{target_stem}{sample.image_path.suffix.lower()}"
            shutil.copy2(sample.image_path, image_target)
            copied_images += 1

            if sample.label_path is not None:
                label_target = labels_out / f"{target_stem}.txt"
                shutil.copy2(sample.label_path, label_target)
                copied_labels += 1
            else:
                (labels_out / f"{target_stem}.txt").write_text("", encoding="utf-8")

    return {
        "prepared_root": str(prepared_root.resolve()),
        "copied_images": copied_images,
        "copied_labels": copied_labels,
        "splits": {split_name: len(samples) for split_name, samples in layout.split_to_samples.items()},
    }
