from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Dict, Any
import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from src.training.losses import get_loss_fn
from src.training.evaluator import evaluate_model
from src.utils.io import save_json

@dataclass
class TrainerConfig:
    epochs: int
    learning_rate: float
    weight_decay: float
    patience: int
    scheduler_factor: float
    scheduler_patience: int
    checkpoint_path: str
    metrics_path: str
    task_mode: str

class Trainer:
    def __init__(self, model, train_loader, val_loader, device, cfg: TrainerConfig):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.cfg = cfg
        self.optimizer = Adam(self.model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode="max", factor=cfg.scheduler_factor, patience=cfg.scheduler_patience)
        self.loss_fn = get_loss_fn(cfg.task_mode)

    def _compute_loss(self, logits, y):
        if self.cfg.task_mode == "binary":
            y = y.float().view(-1, 1)
            return self.loss_fn(logits, y)
        return self.loss_fn(logits, y)

    def train_one_epoch(self) -> float:
        self.model.train()
        losses = []
        for batch in tqdm(self.train_loader, leave=False):
            x = batch["x"].to(self.device)
            y = batch["y"].to(self.device)
            self.optimizer.zero_grad()
            out = self.model(x)
            loss = self._compute_loss(out["logits"], y)
            loss.backward()
            self.optimizer.step()
            losses.append(float(loss.item()))
        return float(np.mean(losses)) if losses else float("nan")

    def fit(self) -> Dict[str, Any]:
        best_score = -1.0
        best_epoch = -1
        bad_epochs = 0
        history = []

        for epoch in range(1, self.cfg.epochs + 1):
            train_loss = self.train_one_epoch()
            val_eval = evaluate_model(self.model, self.val_loader, self.cfg.task_mode, self.device)
            metrics = val_eval["metrics"]
            score = metrics["f1"] if self.cfg.task_mode == "binary" else metrics["macro_f1"]
            self.scheduler.step(score)

            history.append({"epoch": epoch, "train_loss": train_loss, **metrics})

            if score > best_score:
                best_score = score
                best_epoch = epoch
                bad_epochs = 0
                os.makedirs(os.path.dirname(self.cfg.checkpoint_path), exist_ok=True)
                torch.save({"model_state": self.model.state_dict()}, self.cfg.checkpoint_path)
            else:
                bad_epochs += 1
                if bad_epochs >= self.cfg.patience:
                    break

        result = {"best_epoch": best_epoch, "best_score": best_score, "history": history}
        os.makedirs(os.path.dirname(self.cfg.metrics_path), exist_ok=True)
        save_json(result, self.cfg.metrics_path)
        return result
