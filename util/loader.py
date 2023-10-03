from typing import Union
import numpy as np
from numpy.lib.shape_base import expand_dims
import pandas as pd

import torch
import pytorch_lightning as pl


class DataModuleClass(pl.LightningDataModule):
    def __init__(
        self,
        num_workers: int,
        batch_size: int,
        training_data: pd.DataFrame,
        training_label: pd.Series,
        val_data: pd.DataFrame,
        val_label: pd.Series,
        test_data: pd.DataFrame,
        test_label: pd.Series,
    ):
        super().__init__()
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.training_data = training_data
        self.training_label = training_label
        self.validation_data = val_data
        self.validation_label = val_label
        self.testing_data = test_data
        self.testing_label = test_label

    @staticmethod
    def _create_feature_label_tensors(
        features_df: Union[pd.DataFrame, np.array], labels_df: Union[pd.Series, np.array]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(features_df, pd.DataFrame):
            features = features_df.values
        else:
            features = features_df

        if isinstance(labels_df, pd.Series):
            labels = labels_df.values
        else:
            labels = labels_df

        if expand_dims:
            features, labels = np.expand_dims(features, axis=2), np.expand_dims(labels, axis=1)

        return (
            torch.stack([torch.tensor(x, dtype=torch.float) for x in features]),
            torch.tensor(labels, dtype=torch.float),
        )

    def train_dataloader(self) -> "FastTensorDataLoader":
        train_features, train_labels = self._create_feature_label_tensors(
            self.training_data, self.training_label
        )
        return FastTensorDataLoader(
            train_features,
            train_labels,
            batch_size=self.batch_size,
            shuffle=True,
        )

    def val_dataloader(self) -> "FastTensorDataLoader":
        validation_features, validation_labels = self._create_feature_label_tensors(
            self.validation_data, self.validation_label
        )
        return FastTensorDataLoader(
            validation_features,
            validation_labels,
            batch_size=self.batch_size,
            shuffle=False,
        )

    def test_dataloader(self) -> "FastTensorDataLoader":
        test_features, test_labels = self._create_feature_label_tensors(
            self.testing_data, self.testing_label
        )
        return FastTensorDataLoader(
            test_features,
            test_labels,
            batch_size=self.batch_size,
            shuffle=False,
        )


# https://github.com/hcarlens/pytorch-tabular/blob/master/fast_tensor_data_loader.py
class FastTensorDataLoader:
    """
    A DataLoader-like object for a set of tensors that can be much faster than
    TensorDataset + DataLoader because dataloader grabs individual indices of
    the dataset and calls cat (slow). Used to train supervised algorithms. This is modified
    from the original to fit our problem

    Source: https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/27014/6
    """

    def __init__(
        self,
        dataset: torch.Tensor,
        labels: torch.Tensor,
        num_workers: int = 0,
        batch_size: int = 32,
        shuffle: bool = False,
    ):
        """
        Initialize a FastTensorDataLoader.

        :param dataset: Pandas DataFrame of features.
        "param labels: Pandas Series of labels.
        :param batch_size: batch size to load.
        :param num_workers: number of workers to load batches from the dataset.
        :param shuffle: if True, shuffle the data *in-place* whenever an
            iterator is created out of this object.

        :returns: A FastTensorDataLoader.
        """
        if dataset.shape[0] != labels.shape[0]:
            raise ValueError("number of samples for features and labels do not match")

        self.dataset = dataset
        self.labels = labels

        self.dataset_len = self.dataset.shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle

        if num_workers < 0:
            raise ValueError(
                "num_workers option should be non-negative; "
                "use num_workers=0 to disable multiprocessing."
            )
        self.num_workers = num_workers

        # Calculate # batches
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches

    def __iter__(self) -> "FastTensorDataLoader":
        if self.shuffle:
            r = torch.randperm(self.dataset_len)
            self.dataset = self.dataset[r, :]
            self.labels = self.labels[r]
        self.i = 0
        return self

    def __next__(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.i >= self.dataset_len:
            raise StopIteration

        features = self.dataset[self.i : self.i + self.batch_size, :]
        classes = self.labels[self.i : self.i + self.batch_size]
        self.i += self.batch_size
        return features, classes

    def __len__(self) -> int:
        return self.n_batches
