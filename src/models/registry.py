SUPPORTED_MODELS = ("yolo", "faster_rcnn", "ssd", "efficientdet", "detr")

_MODEL_BACKENDS = {
    "yolo": "ultralytics",
    "faster_rcnn": "torchvision",
    "ssd": "torchvision",
    "efficientdet": "effdet",
    "detr": "transformers",
}


def validate_model_name(model_name: str) -> str:
    normalized = model_name.strip().lower()
    if normalized not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unsupported model '{model_name}'. Expected one of: {', '.join(SUPPORTED_MODELS)}."
        )
    return normalized


def get_model_backend(model_name: str) -> str:
    normalized = validate_model_name(model_name)
    return _MODEL_BACKENDS[normalized]

