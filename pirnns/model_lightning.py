import lightning as L
import torch.nn as nn
import torch


class PathIntRNNLightning(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        lr: float = 0.01,
    ) -> None:
        super().__init__()
        self.model = model
        self.lr = lr

        self.train_losses: list[torch.Tensor] = []
        self.val_losses: list[torch.Tensor] = []

    def training_step(self, batch) -> torch.Tensor:
        inputs, targets = batch
        # inputs has shape (batch_size, time_steps, input_size)
        # targets has shape (batch_size, time_steps, output_size)
        hidden_states, outputs = self.model(inputs=inputs, pos_0=targets[:, 0, :])

        loss = nn.functional.mse_loss(outputs, targets)

        self.train_losses.append(loss)
        return loss

    def validation_step(self, batch) -> torch.Tensor:
        inputs, targets = batch
        hidden_states, outputs = self.model(inputs=inputs, pos_0=targets[:, 0, :])

        loss = nn.functional.mse_loss(outputs, targets)

        self.val_losses.append(loss)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)