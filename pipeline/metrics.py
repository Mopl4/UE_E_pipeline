from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


def compute_basic_metrics(*, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, object]:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    f1_macro = float(f1_score(y_true, y_pred, average="macro"))
    f1_weighted = float(f1_score(y_true, y_pred, average="weighted"))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    cm_norm = (cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)).astype(np.float64)
    per_class_err = {}
    for c in [0, 1, 2]:
        mask = y_true == c
        if np.any(mask):
            per_class_err[str(c)] = float(np.mean(y_pred[mask] != c))
    report_dict = classification_report(y_true, y_pred, digits=4, output_dict=True)
    report_text = classification_report(y_true, y_pred, digits=4, output_dict=False)
    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_normalized": cm_norm.tolist(),
        "per_class_error_rate": per_class_err,
        "classification_report": report_text,
        "classification_report_dict": report_dict,
    }
