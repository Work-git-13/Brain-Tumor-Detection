from __future__ import annotations


def build_ssd(num_classes: int):
    import torchvision

    return torchvision.models.detection.ssd300_vgg16(
        weights=None,
        weights_backbone="DEFAULT",
        num_classes=num_classes,
        trainable_backbone_layers=3,
    )
