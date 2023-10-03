import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.container import Sequential

from core.base import BaseModel
from core.layer import Sparsemax, Entmax15
from typing import List
from util.preprocessing import EmbeddingGenerator


def compute_cat_emb_dimensions(modalities: List[int]) -> List[int]:
    """
    Heuristic for calculating number of dimensions needed for a categorical feature embedding
    """
    dims = []
    for value in modalities:
        if value <= 5:
            dims.append(int(value))
        else:
            # taken from a rule of thumb from Google's blog post
            # https://developers.googleblog.com/2017/11/introducing-tensorflow-feature-columns.html
            dims.append(int(round(pow(value, 0.25))))
    return dims


class ConvBlock(nn.Module):
    def __init__(
        self,
        input,
        output,
        kernel_size,
        stride,
        padding,
        dropout: float,
    ):
        super().__init__()
        self.block = Sequential(
            nn.BatchNorm1d(input),
            nn.Dropout(dropout),
            nn.utils.weight_norm(
                nn.Conv1d(
                    input,
                    output,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=False,
                ),
                dim=None,
            ),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.block(x)
        return x


class SparseConvBlock(nn.Module):
    def __init__(self, input, output, kernel_size, stride, padding, sparsemax=True):
        sparse_layer = Sparsemax() if sparsemax else Entmax15()
        super().__init__()
        self.block = Sequential(
            nn.BatchNorm1d(input),
            sparse_layer,
            nn.utils.weight_norm(
                nn.Conv1d(
                    input,
                    output,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=False,
                ),
                dim=None,
            ),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.block(x)
        return x


class LinearBlock(nn.Module):
    def __init__(self, input, output, dropout: float, set_weight_norm: bool = True):
        super().__init__()
        linear_layer = (
            nn.utils.weight_norm(nn.Linear(input, output))
            if set_weight_norm
            else nn.Linear(input, output)
        )
        self.block = Sequential(nn.BatchNorm1d(input), nn.Dropout(dropout), linear_layer)

    def forward(self, x):
        x = self.block(x)
        return x


class SimpleEncoder(BaseModel):
    def __init__(
        self,
        num_features: int,
        num_targets: int,
        cat_idxs: List[int] = [],
        cat_dims: List[int] = [],
        cat_emb_dim: int = 1,
        learning_rate: float = 1e-3,
        decay: float = 1e-5,
        hidden_sizes: List[int] = [128, 64],
    ):
        super().__init__(learning_rate, decay)
        self.embedder = EmbeddingGenerator(num_features, cat_dims, cat_idxs, cat_emb_dim)
        self.post_embed_dim = self.embedder.post_embed_dim
        self.linear1 = LinearBlock(
            self.post_embed_dim, hidden_sizes[0], dropout=0.1, set_weight_norm=True
        )
        self.linear2 = LinearBlock(
            hidden_sizes[0], hidden_sizes[1], dropout=0.1, set_weight_norm=True
        )
        self.linear3 = nn.Linear(hidden_sizes[1], num_targets)
        self.probability = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedder(x)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        x = self.probability(x)
        return x


class SparseCNNEncoder(BaseModel):
    def __init__(
        self,
        num_features: int,
        num_targets: int,
        cat_idxs: List[int] = [],
        cat_dims: List[int] = [],
        cat_emb_dim: int = 1,
        learning_rate: float = 1e-3,
        decay: float = 1e-5,
        hidden_sizes: List[int] = [512, 64, 128, 128],
        sparsemax: bool = True,
    ):
        super().__init__(learning_rate, decay)
        if len(hidden_sizes) != 4:
            return ValueError("hidden sizes list must be length 4")

        hidden_size = hidden_sizes[0]
        cha_1 = hidden_sizes[1]
        cha_2 = hidden_sizes[2]
        cha_3 = hidden_sizes[3]

        cha_1_reshape = int(hidden_size / cha_1)
        cha_po_1 = int(hidden_size / cha_1 / 2)
        cha_po_2 = int(hidden_size / cha_1 / 2 / 2) * cha_3

        self.cha_1 = cha_1
        self.cha_2 = cha_2
        self.cha_3 = cha_3
        self.cha_1_reshape = cha_1_reshape
        self.cha_po_1 = cha_po_1
        self.cha_po_2 = cha_po_2

        self.embedder = EmbeddingGenerator(num_features, cat_dims, cat_idxs, cat_emb_dim)
        self.post_embed_dim = self.embedder.post_embed_dim

        self.linear1 = LinearBlock(
            self.post_embed_dim, hidden_size, dropout=0.1, set_weight_norm=True
        )
        self.block1 = SparseConvBlock(
            cha_1, cha_2, kernel_size=5, stride=1, padding=2, sparsemax=sparsemax
        )
        self.ave_po_c1 = nn.AdaptiveAvgPool1d(output_size=cha_po_1)

        self.block2 = SparseConvBlock(
            cha_2, cha_2, kernel_size=3, stride=1, padding=1, sparsemax=sparsemax
        )
        self.block2_1 = SparseConvBlock(
            cha_2, cha_2, kernel_size=3, stride=1, padding=1, sparsemax=sparsemax
        )
        self.block2_2 = SparseConvBlock(
            cha_2, cha_3, kernel_size=5, stride=1, padding=2, sparsemax=sparsemax
        )

        self.max_po_c2 = nn.MaxPool1d(kernel_size=4, stride=2, padding=1)

        self.flt = nn.Flatten()

        self.linear2 = LinearBlock(cha_po_2, num_targets, dropout=0.1, set_weight_norm=True)
        self.probability = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedder(x)
        x = F.celu(self.linear1(x), alpha=0.06)
        x = x.reshape(x.shape[0], self.cha_1, self.cha_1_reshape)

        x = self.block1(x)

        x = self.ave_po_c1(x)

        x = self.block2(x)
        x_s = x

        x = self.block2_1(x)
        x = self.block2_2(x)
        x = x * x_s

        x = self.max_po_c2(x)

        x = self.flt(x)
        x = self.linear2(x)
        x = self.probability(x)

        return x
