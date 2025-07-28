"""
adapted from: https://github.com/CVLAB-Unibo/nf2vec/blob/main/models/encoder.py
"""

from typing import List
import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], embed_dim: int) -> None:
        super().__init__()

        layers = []
        for idx in range(len(hidden_dims)):
            in_ch = input_dim if idx == 0 else hidden_dims[idx - 1]
            out_ch = hidden_dims[idx]
            layers.append(nn.Conv1d(in_ch, out_ch, 1))
            layers.append(nn.BatchNorm1d(out_ch))
            layers.append(nn.ReLU())

        layers.append(nn.Conv1d(hidden_dims[-1], embed_dim, 1))
        self.layers = nn.Sequential(*layers)
        self.embed_dim = embed_dim

        fc_layers = []
        fc_layers.append(nn.Linear(embed_dim, embed_dim // 2))
        fc_layers.append(nn.ReLU())
        fc_layers.append(nn.Dropout(0.5))
        fc_layers.append(nn.Linear(embed_dim // 2, 2))
        self.fc_layers = fc_layers = nn.Sequential(*fc_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_channels_first = torch.transpose(x, 2, 1)
        x = self.layers(x_channels_first)
        embeds, _ = torch.max(x, 2)  # max for each col, shape (bs, 512)
        logits = self.fc_layers(embeds)  # shape (bs, 2)

        return logits, embeds
