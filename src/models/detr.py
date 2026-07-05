from __future__ import annotations


def build_detr(checkpoint: str, num_classes: int):
    from transformers import DetrForObjectDetection

    return DetrForObjectDetection.from_pretrained(
        checkpoint,
        num_labels=num_classes,
        ignore_mismatched_sizes=True,
    )

