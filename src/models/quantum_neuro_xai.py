from __future__ import annotations
import torch
import torch.nn as nn
from src.models.temporal_cnn import TemporalCNN
from src.models.attention_rnn import AttentionRNN
from src.models.fusion import QuantumClassicalFusion
from src.models.heads import BinaryHead, MultiClassHead
from src.quantum.quantum_block import QuantumFeatureBlock

class QuantumNeuroXAI(nn.Module):
    def __init__(self, model_cfg: dict, task_mode: str = "binary", num_classes: int = 2):
        super().__init__()
        self.task_mode = task_mode
        self.num_classes = num_classes

        q_cfg = model_cfg["quantum"]
        cnn_cfg = model_cfg["cnn"]
        rnn_cfg = model_cfg["rnn"]
        attn_cfg = model_cfg["attention"]
        fus_cfg = model_cfg["fusion"]

        self.cnn = TemporalCNN(
            in_ch=model_cfg.get("input_channels", 1),
            conv_channels=tuple(cnn_cfg["conv_channels"]),
            kernel_size=cnn_cfg["kernel_size"],
            dropout=cnn_cfg["dropout"],
        )

        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        pooled_input_dim = None
        self._quantum_cfg = q_cfg
        self._rnn_cfg = rnn_cfg
        self._attn_cfg = attn_cfg
        self._fus_cfg = fus_cfg

        self.quantum_block = None
        self.rnn = None
        self.fusion = None
        self.head = None
        self._pooled_input_dim = pooled_input_dim

    def _lazy_build(self, x: torch.Tensor):
        with torch.no_grad():
            pooled = self.pool(x).flatten(1)
            feat = self.cnn(x)
            b, c, f, t = feat.shape
            seq_in_dim = c * f

        self.quantum_block = QuantumFeatureBlock(
            input_dim=pooled.shape[1],
            proj_dim=self._quantum_cfg["proj_dim"],
            feature_depth=self._quantum_cfg["feature_depth"],
            measure_dim=self._quantum_cfg["measure_dim"],
            phase_scale=self._quantum_cfg["phase_scale"],
        )
        self.rnn = AttentionRNN(
            input_size=seq_in_dim,
            rnn_type=self._rnn_cfg["type"],
            hidden_size=self._rnn_cfg["hidden_size"],
            num_layers=self._rnn_cfg["num_layers"],
            bidirectional=self._rnn_cfg["bidirectional"],
            dropout=self._rnn_cfg["dropout"],
            attn_hidden=self._attn_cfg["hidden_size"],
        )
        self.fusion = QuantumClassicalFusion(
            quantum_dim=self._quantum_cfg["measure_dim"],
            classical_dim=self.rnn.out_dim,
            hidden_dim=self._fus_cfg["hidden_dim"],
            dropout=self._fus_cfg["dropout"],
        )
        self.head = BinaryHead(self._fus_cfg["hidden_dim"]) if self.task_mode == "binary" else MultiClassHead(self._fus_cfg["hidden_dim"], self.num_classes)

    def forward(self, x: torch.Tensor):
        if self.quantum_block is None:
            self._lazy_build(x)

        pooled = self.pool(x).flatten(1)
        q_vec = self.quantum_block(pooled)

        feat = self.cnn(x)
        b, c, f, t = feat.shape
        seq = feat.permute(0, 3, 1, 2).contiguous().view(b, t, c * f)
        context, attn, seq_out = self.rnn(seq)

        fused = self.fusion(q_vec, context)
        logits = self.head(fused)
        return {
            "logits": logits,
            "probs": torch.sigmoid(logits) if self.task_mode == "binary" else torch.softmax(logits, dim=-1),
            "quantum_vec": q_vec,
            "temporal_context": context,
            "attention_weights": attn,
            "fused_vec": fused,
            "sequence_output": seq_out,
        }
