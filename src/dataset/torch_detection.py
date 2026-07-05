from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.tv_tensors import BoundingBoxes, Image as TVImage
from torchvision.transforms.v2 import functional as F


@dataclass(slots=True)
class TorchDetectionRecord:
    image_path: Path
    label_path: Path


class BrainMRIDetectionDataset(Dataset):
    def __init__(self, dataset_root: str | Path, split: str, class_names: list[str]):
        self.dataset_root = Path(dataset_root)
        self.split = split
        self.class_names = class_names
        self.class_to_index = {name: idx + 1 for idx, name in enumerate(class_names)}
        self.records = self._discover_records()

    def _discover_records(self) -> list[TorchDetectionRecord]:
        image_dir = self.dataset_root / self.split / "images"
        label_dir = self.dataset_root / self.split / "labels"
        records: list[TorchDetectionRecord] = []

        for image_path in sorted(image_dir.glob("*")):
            label_path = label_dir / f"{image_path.stem}.txt"
            records.append(TorchDetectionRecord(image_path=image_path, label_path=label_path))

        return records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        image = Image.open(record.image_path).convert("RGB")
        width, height = image.size

        boxes: list[list[float]] = []
        labels: list[int] = []

        if record.label_path.exists():
            lines = record.label_path.read_text(encoding="utf-8").strip().splitlines()
            for line in lines:
                if not line.strip():
                    continue
                class_id, x_center, y_center, box_width, box_height = map(float, line.split())
                x1 = (x_center - box_width / 2.0) * width
                y1 = (y_center - box_height / 2.0) * height
                x2 = (x_center + box_width / 2.0) * width
                y2 = (y_center + box_height / 2.0) * height
                boxes.append([x1, y1, x2, y2])
                labels.append(int(class_id) + 1)

        image_tensor = F.to_image(image)
        image_tensor = F.to_dtype(image_tensor, torch.float32, scale=True)

        if boxes:
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
            labels_tensor = torch.tensor(labels, dtype=torch.int64)
            area = (boxes_tensor[:, 2] - boxes_tensor[:, 0]) * (boxes_tensor[:, 3] - boxes_tensor[:, 1])
        else:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)
            area = torch.zeros((0,), dtype=torch.float32)

        target = {
            "boxes": BoundingBoxes(boxes_tensor, format="XYXY", canvas_size=(height, width)),
            "labels": labels_tensor,
            "image_id": torch.tensor([index]),
            "area": area,
            "iscrowd": torch.zeros((labels_tensor.shape[0],), dtype=torch.int64),
        }

        return TVImage(image_tensor), target


def detection_collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)
