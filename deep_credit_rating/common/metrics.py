from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_recall_fscore_support,
)

from deep_credit_rating.common import config as cfg


def quadratic_weighted_kappa(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    return float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> dict:
    return {
        "qwk": quadratic_weighted_kappa(y_true, y_pred, num_classes),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "confusion": confusion_matrix(y_true, y_pred, labels=list(range(num_classes))).tolist(),
    }


def compute_extended_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int | None = None) -> dict:
    """
    Metrics cơ bản + chi tiết theo lớp, lớp nguy cơ cao nhất (index num_classes-1),
    MAE trên thang thứ tự, balanced accuracy (trung bình recall theo lớp).
    """
    if num_classes is None:
        num_classes = cfg.NUM_CLASSES
    labels = list(range(num_classes))
    base = compute_metrics(y_true, y_pred, num_classes)
    prec, rec, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    per_class: dict[str, dict] = {}
    for i in labels:
        per_class[str(i)] = {
            "precision": float(prec[i]),
            "recall": float(rec[i]),
            "f1": float(f1[i]),
            "support": int(sup[i]),
        }
    hi = num_classes - 1
    base["per_class"] = per_class
    base["high_risk_class"] = {
        "description_vi": (
            "Lớp nguy cơ cao nhất (chỉ số 0..4; thang 1-5 = index+1). "
            "recall = tỉ lệ nhận diện đúng trong các mẫu thật sự thuộc lớp này."
        ),
        "class_index_0_based": hi,
        "rating_1_to_5": hi + 1,
        "precision": float(prec[hi]),
        "recall": float(rec[hi]),
        "f1": float(f1[hi]),
        "support": int(sup[hi]),
    }
    base["mean_absolute_error"] = float(mean_absolute_error(y_true, y_pred))
    base["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
    return base


def format_confusion_matrix_text(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int | None = None) -> str:
    """Ma trận nhầm lẫn dạng text: hàng = nhãn thật, cột = nhãn dự đoán (0..K-1)."""
    if num_classes is None:
        num_classes = cfg.NUM_CLASSES
    labels = list(range(num_classes))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    lines = [
        "Ma trận nhầm lẫn: hàng = nhãn thật (0..4), cột = nhãn dự đoán (0..4).",
        "rating_1_to_5 = class_index + 1.",
        "",
    ]
    w = max(6, max(len(str(int(x))) for row in cm for x in row) + 1)
    header = "true\\pred".ljust(w) + "".join(str(j).rjust(w) for j in labels)
    lines.append(header)
    for i in labels:
        row = str(i).ljust(w) + "".join(str(int(cm[i, j])).rjust(w) for j in labels)
        lines.append(row)
    return "\n".join(lines)


def format_report(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    return classification_report(y_true, y_pred, zero_division=0)
