import lightning as L
import torch.nn as nn
import torch


class PathIntRNNLightning(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        place_cells,
        lr: float = 0.01,
    ) -> None:
        super().__init__()
        self.model = model
        self.place_cells = place_cells
        self.lr = lr


    def training_step(self, batch) -> torch.Tensor:
        inputs, targets = batch
        # inputs has shape (batch_size, time_steps, input_size)
        # targets has shape (batch_size, time_steps, output_size)
        hidden_states, outputs = self.model(inputs=inputs, pos_0=targets[:, 0, :])

        # Convert true positions to place cell activations
        pc_targets = self.place_cells.get_activation(targets)

        # Negative log likelihood loss
        loss = -(pc_targets * torch.log_softmax(outputs, dim=-1)).sum(-1).mean()

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        
        return loss

    def validation_step(self, batch) -> torch.Tensor:
        inputs, targets = batch
        hidden_states, outputs = self.model(inputs=inputs, pos_0=targets[:, 0, :])

        pc_targets = self.place_cells.get_activation(targets)

        loss = -(pc_targets * torch.log_softmax(outputs, dim=-1)).sum(-1).mean()

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)