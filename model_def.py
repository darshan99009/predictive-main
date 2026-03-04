"""
model_def.py
LSTM architecture — must exactly match the training code.
Imported by app.py to load saved .pt weights.
"""

import torch
import torch.nn as nn

HIDDEN_SIZE = 128
NUM_LAYERS  = 2
DROPOUT     = 0.2


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, lstm_out):
        # lstm_out: (batch, seq, hidden)
        weights = torch.softmax(self.attn(lstm_out), dim=1)  # (batch, seq, 1)
        return (weights * lstm_out).sum(dim=1)                # (batch, hidden)


class LSTMModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.lstm      = nn.LSTM(input_size, HIDDEN_SIZE, NUM_LAYERS,
                                 batch_first=True, dropout=DROPOUT)
        self.attention = Attention(HIDDEN_SIZE)
        self.bn        = nn.BatchNorm1d(HIDDEN_SIZE)
        self.dropout   = nn.Dropout(0.3)
        self.fc1       = nn.Linear(HIDDEN_SIZE, 64)
        self.fc2       = nn.Linear(64, 1)
        self.relu      = nn.ReLU()

    def forward(self, x):
        out, _  = self.lstm(x)
        context = self.attention(out)
        context = self.dropout(self.bn(context))
        return self.fc2(self.relu(self.fc1(context))).squeeze(-1)
