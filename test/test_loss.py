import math
import pytest
import torch

from core.loss import HazardLoss


def test_hazard_loss_eps_failure():
    with pytest.raises(ValueError, match="eps value"):
        HazardLoss(eps=0.0)


def test_hazard_loss_target_dimension_failure():
    target_tensor = torch.Tensor([0])
    input_tensor = torch.Tensor([1])

    with pytest.raises(ValueError):
        loss = HazardLoss()
        loss.forward(input_tensor, target_tensor)


def test_hazard_loss_final_target_dimension_failure():
    target_tensor = torch.Tensor([[[1], [12]], [[5], [1]]])
    input_tensor = torch.Tensor([[[0.1], [0.9]], [[0.85], [0.05]]])

    with pytest.raises(ValueError):
        loss = HazardLoss()
        loss.forward(input_tensor, target_tensor)


def test_smoke():
    target_tensor = torch.Tensor([[[1, 1], [12, 0]], [[5, 0], [3, 1]]])
    input_tensor = torch.Tensor([[[0.1], [0.9]], [[0.85], [0.05]]])

    eps = 1e-5
    loss = HazardLoss(eps=eps)
    loss_value = loss.forward(input_tensor, target_tensor)
    assert math.isclose(loss_value.item(), 5.75888, rel_tol=1e-5)


def test_single_output_retained():
    target_tensor = torch.Tensor([[[1, 0]]])
    input_tensor = torch.Tensor([[[0.5]]])

    eps = 1e-5
    loss = HazardLoss(eps=eps)
    loss_value = loss.forward(input_tensor, target_tensor)
    assert math.isclose(loss_value.item(), -1.0 * math.log(0.5 + eps), rel_tol=1e-3)


def test_single_output_churned():
    target_tensor = torch.Tensor([[[2, 1]]])
    input_tensor = torch.Tensor([[[0.5]]])

    eps = 1e-5
    loss = HazardLoss(eps=eps)
    loss_value = loss.forward(input_tensor, target_tensor)
    assert math.isclose(
        loss_value.item(), -1.0 * math.log(0.5 + eps) - 1.0 * math.log(0.5 + eps), rel_tol=1e-3
    )
