from __future__ import annotations


def build_faster_rcnn(num_classes: int):
    import torchvision
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

