from __future__ import annotations
from typing import Dict, Any
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
from src.training.metrics import compute_binary_metrics, compute_multiclass_metrics

@torch.no_grad()
def evaluate_model(model, loader, task_mode: str, device: torch.device) -> Dict[str, Any]:
    model.eval()
    y_true, y_prob, y_pred = [], [], []

    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        out = model(x)
        logits = out["logits"]
        if task_mode == "binary":
            prob = torch.sigmoid(logits).view(-1).cpu().numpy()
            pred = (prob >= 0.5).astype(int)
            y_prob.extend(prob.tolist())
            y_pred.extend(pred.tolist())
            y_true.extend(y.cpu().numpy().tolist())
        else:
            prob = torch.softmax(logits, dim=-1).cpu().numpy()
            pred = prob.argmax(axis=1)
            y_prob.extend(prob.tolist())
            y_pred.extend(pred.tolist())
            y_true.extend(y.cpu().numpy().tolist())

    if task_mode == "binary":
        metrics = compute_binary_metrics(y_true, y_prob)
    else:
        metrics = compute_multiclass_metrics(y_true, y_pred, y_prob=np.asarray(y_prob))
    return {"metrics": metrics, "y_true": y_true, "y_pred": y_pred, "y_prob": y_prob}

def export_confusion_matrix(y_true, y_pred, save_path: str, title: str = "Confusion Matrix") -> None:
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("True")
    for (i, j), val in np.ndenumerate(cm):
        plt.text(j, i, int(val), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def export_roc_curve(y_true, y_prob, save_path: str, title: str = "ROC Curve") -> None:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
