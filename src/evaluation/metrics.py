from __future__ import annotations


def summarize_detection_metrics(metrics: dict) -> str:
    ordered_keys = ("map50", "map50_95", "precision", "recall", "f1")
    available = [f"{key}={metrics[key]:.4f}" for key in ordered_keys if key in metrics]
    return ", ".join(available) if available else "No metrics available."

