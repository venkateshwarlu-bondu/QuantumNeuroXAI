from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn

def get_loss_fn(task_mode: str, class_weights: Optional[torch.Tensor] = None):
    if task_mode == "binary":
        return nn.BCEWithLogitsLoss()
    if class_weights is not None:
        return nn.CrossEntropyLoss(weight=class_weights)
    return nn.CrossEntropyLoss()
