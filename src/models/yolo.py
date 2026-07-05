from __future__ import annotations


def build_yolo_model(weights: str):
    from ultralytics import YOLO

    return YOLO(weights)

