import pytest
import torch


@pytest.fixture(scope="class")
def test_dataset():
    return torch.Tensor([[1, 2, 3], [4, 5, 6]])
