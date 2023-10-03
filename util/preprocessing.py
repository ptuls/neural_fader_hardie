import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from typing import List, Union


class StandardScaler:
    def __init__(self, tol: float = 1e-7) -> None:
        self.tol = tol
        self.check_fitted = False

    def fit(self, x: torch.Tensor) -> None:
        self.mean = x.mean(0, keepdim=True)
        self.std = x.std(0, unbiased=False, keepdim=True)
        self.check_fitted = True

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        if self.check_fitted:
            x -= self.mean
            x /= self.std + self.tol
            return x
        else:
            raise RuntimeError("standard scaler not fitted yet")


class CategoricalReconstructor:
    def __init__(self, prefixes: List[str] = []) -> None:
        self.prefixes = prefixes

    def fit(self, *args, **kwargs) -> "CategoricalReconstructor":
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        categoricals = {}
        for prefix in self.prefixes:
            selected = x.columns[x.columns.str.startswith(prefix)]
            categoricals[prefix] = x[selected].idxmax(axis=1)

        return pd.DataFrame(categoricals)


# Original from https://dreamquark-ai.github.io/tabnet/_modules/pytorch_tabnet/tab_network.html#EmbeddingGenerator
class EmbeddingGenerator(nn.Module):
    """
    Classical embeddings generator
    """

    def __init__(
        self,
        input_dim: int,
        cat_dims: List[int],
        cat_idxs: List[int],
        cat_emb_dim: Union[List[int], int],
    ):
        """This is an embedding module for an entire set of features
        Parameters
        ----------
        input_dim : int
            Number of features coming as input (number of columns)
        cat_dims : list of int
            Number of modalities for each categorical features
            If the list is empty, no embeddings will be done
        cat_idxs : list of int
            Positional index for each categorical features in inputs
        cat_emb_dim : int or list of int
            Embedding dimension for each categorical features
            If int, the same embedding dimension will be used for all categorical features
        """
        super().__init__()
        if cat_dims == [] and cat_idxs == []:
            self.skip_embedding = True
            self.post_embed_dim = input_dim
            return
        elif (cat_dims == []) or (cat_idxs == []):
            if cat_dims == []:
                msg = "if cat_idxs is non-empty, cat_dims must be defined as a list of same length"
            else:
                msg = "if cat_dims is non-empty, cat_idxs must be defined as a list of same length"
            raise ValueError(msg)
        elif len(cat_dims) != len(cat_idxs):
            msg = "the lists cat_dims and cat_idxs must have the same length"
            raise ValueError(msg)

        self.skip_embedding = False
        if isinstance(cat_emb_dim, int):
            self.cat_emb_dims = [cat_emb_dim] * len(cat_idxs)
        else:
            self.cat_emb_dims = cat_emb_dim

        # check that all embeddings are provided
        if len(self.cat_emb_dims) != len(cat_dims):
            msg = f"""cat_emb_dim and cat_dims must be lists of same length, got {len(self.cat_emb_dims)}
                      and {len(cat_dims)}"""
            raise ValueError(msg)
        self.post_embed_dim = int(input_dim + np.sum(self.cat_emb_dims) - len(self.cat_emb_dims))

        self.embeddings = torch.nn.ModuleList()

        # Sort dims by cat_idx
        sorted_idxs = np.argsort(cat_idxs)
        cat_dims = [cat_dims[i] for i in sorted_idxs]
        self.cat_emb_dims = [self.cat_emb_dims[i] for i in sorted_idxs]

        for cat_dim, emb_dim in zip(cat_dims, self.cat_emb_dims):
            self.embeddings.append(torch.nn.Embedding(cat_dim, emb_dim))

        # record continuous indices
        self.continuous_idx = torch.ones(input_dim, dtype=torch.bool)
        self.continuous_idx[cat_idxs] = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply embeddings to inputs
        Inputs should be (batch_size, input_dim)
        Outputs will be of size (batch_size, self.post_embed_dim)
        """
        if self.skip_embedding:
            # no embeddings required
            return x

        cols = []
        cat_feat_counter = 0
        for feat_init_idx, is_continuous in enumerate(self.continuous_idx):
            # Enumerate through continuous idx boolean mask to apply embeddings
            if is_continuous:
                cols.append(x[:, feat_init_idx].float().view(-1, 1))
            else:
                cols.append(
                    torch.squeeze(self.embeddings[cat_feat_counter](x[:, feat_init_idx].long()))
                )
                cat_feat_counter += 1
        # concat
        post_embeddings = torch.cat(cols, dim=1)
        return post_embeddings
