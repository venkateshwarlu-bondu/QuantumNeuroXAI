from __future__ import annotations
import torch
import torch.nn as nn
from src.models.temporal_cnn import TemporalCNN
from src.models.attention_rnn import AttentionRNN
from src.models.heads import BinaryHead, MultiClassHead

class BaselineEEGNet(nn.Module):
    def __init__(self, model_cfg: dict, task_mode: str = "binary", num_classes: int = 2):
        super().__init__()
        cnn_cfg = model_cfg["cnn"]
        rnn_cfg = model_cfg["rnn"]
        attn_cfg = model_cfg["attention"]

        self.cnn = TemporalCNN(
            in_ch=model_cfg.get("input_channels", 1),
            conv_channels=tuple(cnn_cfg["conv_channels"]),
            kernel_size=cnn_cfg["kernel_size"],
            dropout=cnn_cfg["dropout"],
        )
        self.rnn = None
        self.task_mode = task_mode

        # Lazy init after seeing tensor shape
        self._rnn_cfg = rnn_cfg
        self._attn_cfg = attn_cfg
        self._head = None
        self._feature_dim = None
        self.num_classes = num_classes

    def _build_seq_modules(self, x: torch.Tensor):
        with torch.no_grad():
            feat = self.cnn(x)
            b, c, f, t = feat.shape
            seq_in_dim = c * f
        self.rnn = AttentionRNN(
            input_size=seq_in_dim,
            rnn_type=self._rnn_cfg["type"],
            hidden_size=self._rnn_cfg["hidden_size"],
            num_layers=self._rnn_cfg["num_layers"],
            bidirectional=self._rnn_cfg["bidirectional"],
            dropout=self._rnn_cfg["dropout"],
            attn_hidden=self._attn_cfg["hidden_size"],
        )
        self._feature_dim = self.rnn.out_dim
        self._head = BinaryHead(self._feature_dim) if self.task_mode == "binary" else MultiClassHead(self._feature_dim, self.num_classes)

    def forward(self, x: torch.Tensor):
        feat = self.cnn(x)
        b, c, f, t = feat.shape
        seq = feat.permute(0, 3, 1, 2).contiguous().view(b, t, c * f)
        if self.rnn is None:
            self._build_seq_modules(x)
        context, attn, seq_out = self.rnn(seq)
        logits = self._head(context)
        return {
            "logits": logits,
            "probs": torch.sigmoid(logits) if self.task_mode == "binary" else torch.softmax(logits, dim=-1),
            "attention_weights": attn,
            "fused_vec": context,
            "sequence_output": seq_out,
        }
