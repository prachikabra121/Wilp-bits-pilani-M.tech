# Common function to calculate the six evaluation metrics.
# It is used by both train_models.py and app.py so the numbers always match.
import numpy as np
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef)


def compute_metrics(y_true, y_pred, y_proba):
    n_classes = len(np.unique(y_true))

    # binary problem: use the default scoring, multi-class: weighted average
    avg = "binary" if n_classes == 2 else "weighted"

    if n_classes == 2:
        auc = roc_auc_score(y_true, y_proba[:, 1])
    else:
        auc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted")

    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": auc,
        "Precision": precision_score(y_true, y_pred, average=avg, zero_division=0),
        "Recall": recall_score(y_true, y_pred, average=avg, zero_division=0),
        "F1": f1_score(y_true, y_pred, average=avg, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }
