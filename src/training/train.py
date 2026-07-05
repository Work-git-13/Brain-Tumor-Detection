from __future__ import annotations

from pathlib import Path

import torch

from src.dataset import collect_dataset_statistics, discover_dataset_layout, prepare_flat_yolo_dataset
from src.models import get_model_backend, validate_model_name
from src.utils.io import write_json
from src.utils.paths import project_root, resolve_from_root

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable


def run(config: dict, model_name: str, mode: str = "plan") -> None:
    model_name = validate_model_name(model_name)
    dataset_root = Path(config["data"]["dataset_root"])
    layout = discover_dataset_layout(dataset_root, config["data"]["split_names"])

    print(f"Project: {config['project']['name']}")
    print(f"Model: {model_name}")
    print(f"Backend: {get_model_backend(model_name)}")
    print(f"Dataset root: {dataset_root.resolve()}")
    print(f"Classes: {layout.class_names or config['data']['class_names']}")
    print(f"Images discovered: {layout.total_images}")

    for split_name, samples in layout.split_to_samples.items():
        print(f"  - {split_name}: {len(samples)} images")

    if mode == "plan":
        print("Plan mode: dataset inspection completed, training was not started.")
        return

    if mode == "prepare":
        summary = prepare_flat_yolo_dataset(config)
        write_json(summary, project_root() / "results" / "logs" / "prepare_summary.json")
        print(f"Prepared dataset root: {summary['prepared_root']}")
        print(f"Copied images: {summary['copied_images']}")
        print(f"Copied labels: {summary['copied_labels']}")
        for split_name, count in summary["splits"].items():
            print(f"  - {split_name}: {count} samples")
        return

    if mode == "analyze":
        stats = collect_dataset_statistics(dataset_root, config["data"]["split_names"])
        write_json(stats, project_root() / "results" / "logs" / "dataset_summary.json")
        print(f"Total images: {stats['total_images']}")
        print(f"Total boxes: {stats['total_boxes']}")
        print(f"Missing labels: {stats['missing_labels']}")
        print(f"Empty labels: {stats['empty_labels']}")
        for split_name, split_stats in stats["splits"].items():
            print(f"[{split_name}] images={split_stats['images']} boxes={split_stats['boxes']}")
            print(f"  per_class_images={split_stats['per_class_images']}")
            print(f"  per_class_boxes={split_stats['per_class_boxes']}")
        return

    if mode != "train":
        raise ValueError("Mode must be one of: 'plan', 'analyze', 'prepare', 'train'.")

    if model_name == "yolo":
        _train_yolo(config)
        return

    if model_name in {"faster_rcnn", "ssd"}:
        _train_torchvision_detector(config, model_name)
        return

    raise NotImplementedError(_build_pending_model_message(model_name))


def _train_yolo(config: dict) -> None:
    from src.models.yolo import build_yolo_model
    from src.utils.paths import build_yolo_data_yaml

    data_yaml = build_yolo_data_yaml(config)
    model = build_yolo_model(config["models"]["yolo"]["weights"])
    train_cfg = config["training"]
    yolo_cfg = config["models"]["yolo"]

    model.train(
        data=str(data_yaml),
        epochs=train_cfg["epochs"],
        imgsz=yolo_cfg["imgsz"],
        batch=config["data"]["batch_size"],
        project=train_cfg["save_dir"],
        name="yolo_baseline",
    )


def _train_torchvision_detector(config: dict, model_name: str) -> None:
    from src.training.dataloaders import build_detection_dataloader

    device = _resolve_device(config)
    train_loader = build_detection_dataloader(config, split="train")
    val_loader = build_detection_dataloader(config, split="val")
    num_classes = len(config["data"]["class_names"]) + 1
    model = _build_torchvision_model(model_name, num_classes).to(device)

    train_cfg = config["training"]
    optimizer = _build_optimizer(train_cfg, model)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=max(1, train_cfg["epochs"] // 3),
        gamma=0.1,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=bool(train_cfg.get("mixed_precision")) and device.type == "cuda")

    model_results_dir = resolve_from_root(train_cfg["save_dir"]) / model_name
    model_results_dir.mkdir(parents=True, exist_ok=True)
    model_artifacts_dir = project_root() / "artifacts" / model_name
    model_artifacts_dir.mkdir(parents=True, exist_ok=True)

    history: list[dict] = []
    best_score = float("-inf")
    best_weights_path = model_artifacts_dir / f"best_{model_name}.pt"
    last_weights_path = model_artifacts_dir / f"last_{model_name}.pt"

    print(f"Training {model_name} on {device} for {train_cfg['epochs']} epochs")
    for epoch in range(1, train_cfg["epochs"] + 1):
        train_loss = _train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            mixed_precision=bool(train_cfg.get("mixed_precision")),
            model_name=model_name,
            epoch=epoch,
            total_epochs=train_cfg["epochs"],
        )
        metrics = _evaluate_detector(
            model=model,
            loader=val_loader,
            device=device,
            score_threshold=float(config["evaluation"]["score_threshold"]),
        )
        scheduler.step()

        epoch_row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "map": metrics["map"],
            "map_50": metrics["map_50"],
            "mar_100": metrics["mar_100"],
        }
        history.append(epoch_row)
        print(
            f"Epoch {epoch}/{train_cfg['epochs']} | "
            f"loss={train_loss:.4f} map={metrics['map']:.4f} "
            f"map_50={metrics['map_50']:.4f} mar_100={metrics['mar_100']:.4f}"
        )

        if metrics["map_50"] >= best_score:
            best_score = metrics["map_50"]
            torch.save(model.state_dict(), best_weights_path)
            write_json(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    **metrics,
                    "weights_path": str(best_weights_path),
                },
                model_results_dir / "best_metrics.json",
            )

    torch.save(model.state_dict(), last_weights_path)
    import pandas as pd

    pd.DataFrame(history).to_csv(model_results_dir / "metrics.csv", index=False)
    print(f"Best weights: {best_weights_path}")
    print(f"Last weights: {last_weights_path}")
    print(f"Metrics CSV: {model_results_dir / 'metrics.csv'}")


def _build_pending_model_message(model_name: str) -> str:
    messages = {
        "detr": (
            "Training for 'detr' is not wired yet. "
            "The project currently uses the dedicated Colab notebook "
            "'notebooks/colab_detr_brain_mri.ipynb' for this model."
        ),
        "efficientdet": (
            "Training for 'efficientdet' is not wired yet. "
            "The project currently uses the dedicated Colab notebook "
            "'notebooks/colab_efficientdet_brain_mri.ipynb' for this model."
        ),
    }
    return messages.get(
        model_name,
        f"Training for '{model_name}' is scaffolded but not wired yet.",
    )


def _build_torchvision_model(model_name: str, num_classes: int):
    if model_name == "faster_rcnn":
        from src.models.faster_rcnn import build_faster_rcnn

        return build_faster_rcnn(num_classes)

    if model_name == "ssd":
        from src.models.ssd import build_ssd

        return build_ssd(num_classes)

    raise ValueError(f"Unsupported torchvision detector: {model_name}")


def _resolve_device(config: dict) -> torch.device:
    requested_device = str(config["project"].get("device", "auto")).lower()
    if requested_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested_device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable, falling back to CPU.")
        return torch.device("cpu")
    return torch.device(requested_device)


def _build_optimizer(train_cfg: dict, model) -> torch.optim.Optimizer:
    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer_name = str(train_cfg.get("optimizer", "adamw")).lower()

    if optimizer_name == "sgd":
        return torch.optim.SGD(
            params,
            lr=float(train_cfg["learning_rate"]),
            momentum=float(train_cfg.get("momentum", 0.9)),
            weight_decay=float(train_cfg["weight_decay"]),
        )

    if optimizer_name == "adamw":
        return torch.optim.AdamW(
            params,
            lr=float(train_cfg["learning_rate"]),
            weight_decay=float(train_cfg["weight_decay"]),
        )

    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def _train_one_epoch(
    model,
    loader,
    optimizer,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler,
    mixed_precision: bool,
    model_name: str,
    epoch: int,
    total_epochs: int,
) -> float:
    model.train()
    running_loss = 0.0
    progress = tqdm(loader, desc=f"{model_name} train {epoch}/{total_epochs}", leave=False)

    for images, targets in progress:
        images = [image.to(device) for image in images]
        targets = [_move_target_to_device(target, device) for target in targets]

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=mixed_precision and device.type == "cuda"):
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        progress.set_postfix(loss=f"{loss.item():.4f}")

    return running_loss / max(1, len(loader))


def _evaluate_detector(model, loader, device: torch.device, score_threshold: float) -> dict:
    from torchmetrics.detection.mean_ap import MeanAveragePrecision

    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")
    model.eval()

    with torch.no_grad():
        for images, targets in tqdm(loader, desc="val", leave=False):
            images = [image.to(device) for image in images]
            outputs = model(images)
            preds = [_prepare_prediction_for_metric(output, score_threshold) for output in outputs]
            refs = [_prepare_target_for_metric(target) for target in targets]
            metric.update(preds, refs)

    summary = metric.compute()
    return {
        "map": float(summary["map"].item()),
        "map_50": float(summary["map_50"].item()),
        "mar_100": float(summary["mar_100"].item()),
    }


def _move_target_to_device(target: dict, device: torch.device) -> dict:
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in target.items()}


def _prepare_prediction_for_metric(prediction: dict, score_threshold: float) -> dict:
    scores = prediction["scores"].detach().cpu()
    keep = scores >= score_threshold
    return {
        "boxes": prediction["boxes"].detach().cpu()[keep],
        "scores": scores[keep],
        "labels": prediction["labels"].detach().cpu()[keep],
    }


def _prepare_target_for_metric(target: dict) -> dict:
    return {
        "boxes": target["boxes"].detach().cpu(),
        "labels": target["labels"].detach().cpu(),
    }
