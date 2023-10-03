import pytorch_lightning as pl
import torch

from core.loss import HazardLoss


class BaseModel(pl.LightningModule):
    def __init__(self, learning_rate: float, weight_decay: float, eps: float = 1e-6):
        super().__init__()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.eps = eps
        self.loss = HazardLoss()

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        return loss

    def test_step(self, batch, batch_idx) -> torch.Tensor:
        x, y = batch
        y_hat = self.forward(x)

        return {"test_loss": self.loss(y_hat, y)}

    def test_epoch_end(self, outputs) -> torch.Tensor:
        avg_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        logs = {"test_loss": avg_loss}
        results = {
            "avg_test_loss": avg_loss,
            "log": logs,
            "progress_bar": logs,
        }
        self.test_results = results
        return results

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )

    def predict(self, x: torch.Tensor, period: int) -> torch.Tensor:
        if period <= 0:
            return ValueError("period needs to be a positive integer")

        p = self.forward(x)
        return torch.exp(
            torch.tensor(period - 1, dtype=float) * torch.log(1 - p + self.eps)
            + torch.log(p + self.eps)
        )
