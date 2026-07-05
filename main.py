from __future__ import annotations

import argparse

from src.training import run
from src.utils import load_yaml_config, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Brain MRI tumor detection experiments")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config")
    parser.add_argument(
        "--model",
        default="yolo",
        choices=["yolo", "faster_rcnn", "ssd", "efficientdet", "detr"],
        help="Model to inspect or train",
    )
    parser.add_argument(
        "--mode",
        default="plan",
        choices=["plan", "analyze", "prepare", "train"],
        help="Use plan/analyze/prepare before train to validate dataset and experiment setup",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    seed_everything(int(config["project"]["seed"]))
    run(config=config, model_name=args.model, mode=args.mode)


if __name__ == "__main__":
    main()
