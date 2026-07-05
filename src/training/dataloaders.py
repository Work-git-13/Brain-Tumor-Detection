from __future__ import annotations

from torch.utils.data import DataLoader

from src.dataset import BrainMRIDetectionDataset, detection_collate_fn


def build_detection_dataloader(config: dict, split: str) -> DataLoader:
    dataset = BrainMRIDetectionDataset(
        dataset_root=config["data"]["prepared_root"],
        split=split,
        class_names=config["data"]["class_names"],
    )

    return DataLoader(
        dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=split == "train",
        num_workers=config["data"]["num_workers"],
        collate_fn=detection_collate_fn,
    )
