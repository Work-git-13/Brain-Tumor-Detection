from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass(slots=True)
class DetectionSample:
    image_path: Path
    label_path: Path | None
    class_name: str
    split_name: str


@dataclass(slots=True)
class YOLODatasetLayout:
    dataset_root: Path
    split_to_samples: dict[str, list[DetectionSample]]
    class_names: list[str]

    @property
    def total_images(self) -> int:
        return sum(len(samples) for samples in self.split_to_samples.values())


def discover_dataset_layout(dataset_root: str | Path, split_names: dict[str, str]) -> YOLODatasetLayout:
    root = Path(dataset_root)
    split_to_samples: dict[str, list[DetectionSample]] = {}
    class_names: list[str] = []

    for split_key, split_dir_name in split_names.items():
        split_dir = root / split_dir_name
        samples: list[DetectionSample] = []

        if not split_dir.exists():
            split_to_samples[split_key] = samples
            continue

        for class_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
            if class_dir.name not in class_names:
                class_names.append(class_dir.name)

            images_dir = class_dir / "images"
            labels_dir = class_dir / "labels"
            if not images_dir.exists():
                continue

            for image_path in sorted(images_dir.iterdir()):
                if image_path.suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                label_path = labels_dir / f"{image_path.stem}.txt"
                samples.append(
                    DetectionSample(
                        image_path=image_path,
                        label_path=label_path if label_path.exists() else None,
                        class_name=class_dir.name,
                        split_name=split_key,
                    )
                )

        split_to_samples[split_key] = samples

    return YOLODatasetLayout(
        dataset_root=root,
        split_to_samples=split_to_samples,
        class_names=class_names,
    )

