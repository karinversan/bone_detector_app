from __future__ import annotations


def format_metric(value: object) -> str:
    if value is None:
        return "not set"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def metrics_with_f1(metrics: dict) -> dict:
    derived = dict(metrics)
    precision = metrics.get("precision")
    recall = metrics.get("recall")
    if precision is None:
        precision = metrics.get("mp")
    if recall is None:
        recall = metrics.get("mr")
    if precision is None:
        precision = metrics.get("metrics/precision(B)")
    if recall is None:
        recall = metrics.get("metrics/recall(B)")
    if isinstance(precision, (int, float)) and isinstance(recall, (int, float)):
        if precision + recall > 0:
            derived["f1 (computed)"] = (2 * precision * recall) / (precision + recall)
    return derived


def dict_to_rows(data: dict, key_name: str, value_name: str) -> list[dict]:
    rows = []
    for key, value in data.items():
        rows.append({key_name: key, value_name: value})
    return rows


def dict_to_rows_str(data: dict, key_name: str, value_name: str) -> list[dict]:
    rows = []
    for key, value in data.items():
        rows.append({key_name: str(key), value_name: str(value)})
    return rows


def metrics_to_display(metrics: dict) -> dict:
    return {key: format_metric(value) for key, value in metrics.items()}
