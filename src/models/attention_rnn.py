from __future__ import annotations
import torch
import torch.nn as nn

class AttentionLayer(nn.Module):
    def __init__(self, in_dim: int, hidden_size: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, hidden_size)
        self.score = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor):
        # x: [batch, seq, feat]
        h = torch.tanh(self.fc(x))
        scores = self.score(h).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        context = torch.sum(x * weights.unsqueeze(-1), dim=1)
        return context, weights

class AttentionRNN(nn.Module):
    def __init__(self, input_size: int, rnn_type: str = "gru", hidden_size: int = 128, num_layers: int = 1, bidirectional: bool = True, dropout: float = 0.1, attn_hidden: int = 128):
        super().__init__()
        rnn_cls = nn.GRU if rnn_type.lower() == "gru" else nn.LSTM
        self.rnn = rnn_cls(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        out_dim = hidden_size * (2 if bidirectional else 1)
        self.attn = AttentionLayer(out_dim, attn_hidden)
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor):
        out, _ = self.rnn(x)
        context, weights = self.attn(out)
        return context, weights, out
