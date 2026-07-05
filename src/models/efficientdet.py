from __future__ import annotations


def build_efficientdet(variant: str, num_classes: int):
    from effdet import create_model

    return create_model(variant, bench_task="train", num_classes=num_classes)

