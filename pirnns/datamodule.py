import lightning as L
from torch.utils.data import DataLoader
import torch
import math
from torch.utils.data import TensorDataset, random_split



class PathIntegrationDataModule(L.LightningDataModule):
    def __init__(
        self,
        num_trajectories: int,
        batch_size: int,
        num_workers: int,
        train_val_split: float,
        start_time: float,
        end_time: float,
        num_time_steps: int,
        box_width: float = 2.2, #changed arena_L to box_width and box_height, value changed from 5 to 2.2
        box_height: float = 2.2,
        mu_speed: float = 1,
        sigma_speed: float = 0.5,
        tau_vel: float = 1,
        fixed_start: bool = True,  
    ) -> None:
        super().__init__()
        self.num_trajectories = num_trajectories
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_val_split = train_val_split
        self.start_time = start_time
        self.end_time = end_time
        self.num_time_steps = num_time_steps
        self.dt = (end_time - start_time) / num_time_steps

        self.box_width = box_width #changed arena_L to box_width and box_height
        self.box_height = box_height
        self.mu_speed = mu_speed
        self.sigma_speed = sigma_speed
        self.tau_vel = tau_vel
        self.fixed_start = fixed_start
        self._start_pos_tensor = None
    def _simulate_trajectories(
        self,
        device: str = "cpu",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Simulates a batch of trajectories following the Ornstein-Uhlenbeck process.
        (Brownian motion with a drift term).

        Parameters
        ----------
        device : str
            The device to use for the simulation.
        Returns
        -------
        x         : (batch, T, 2), [heading, speed] at each time step
        positions : (batch, T, 2), ground-truth (x,y) positions (optional target)
        """
        # --- initial position & velocity ----------------------------------------

        half_w = self.box_width / 2.0
        half_h = self.box_height / 2.0

        if self.fixed_start:
            pos = self._start_pos_tensor.to(device).expand(self.num_trajectories, 2).clone()
        else:
            pos_x = (torch.rand(self.num_trajectories, device=device) - 0.5) * self.box_width
            pos_y = (torch.rand(self.num_trajectories, device=device) - 0.5) * self.box_height
            pos = torch.stack((pos_x, pos_y), dim=-1)

        # sample initial heading uniformly in (0, 2pi), speed around mu_speed
        hd0 = torch.rand(self.num_trajectories, device=device) * 2 * torch.pi
        spd0 = torch.clamp(
            torch.randn(self.num_trajectories, device=device) * self.sigma_speed
            + self.mu_speed,
            min=0.0,
        )
        vel = torch.stack((torch.cos(hd0), torch.sin(hd0)), dim=-1) * spd0.unsqueeze(-1)

        pos_all, vel_all = [pos], [vel]

        sqrt_2dt_over_tau = math.sqrt(2 * self.dt / self.tau_vel)
        for _ in range(self.num_time_steps - 1):
            # OU velocity update (momentum)
            noise = torch.randn_like(vel)
            vel = (
                vel
                + (self.dt / self.tau_vel) * (-vel)
                + self.sigma_speed * sqrt_2dt_over_tau * noise
            )

            # position update
            pos = pos + vel * self.dt

            # --- reflective boundaries -----------------------------------------
            out_left = pos[:, 0] < -half_w
            out_right = pos[:, 0] > half_w
            out_bottom = pos[:, 1] < -half_h
            out_top = pos[:, 1] > half_h

            # reflect positions and flip corresponding velocity component
            if out_left.any():
                # reflect across x = -half_w
                pos[out_left, 0] = -2 * half_w - pos[out_left, 0]
                vel[out_left, 0] *= -1
            if out_right.any():
                pos[out_right, 0] = 2 * half_w - pos[out_right, 0]
                vel[out_right, 0] *= -1
            if out_bottom.any():
                pos[out_bottom, 1] = -2 * half_h - pos[out_bottom, 1]
                vel[out_bottom, 1] *= -1
            if out_top.any():
                pos[out_top, 1] = 2 * half_h - pos[out_top, 1]
                vel[out_top, 1] *= -1

            pos_all.append(pos)
            vel_all.append(vel)

        vel_all = torch.stack(vel_all, 1)  # (batch, T, 2)
        pos_all = torch.stack(pos_all, 1)  # (batch, T, 2)
        speeds = torch.linalg.norm(vel_all, dim=-1)
        headings = torch.atan2(vel_all[..., 1], vel_all[..., 0]) % (2 * torch.pi)

        input = torch.stack((headings, speeds), dim=-1)  # (batch, T, 2)
        return input, pos_all

    def setup(self, stage=None) -> None:
        if self.fixed_start:
            # sample one random start once
            x0 = (torch.rand(1).item() - 0.5) * self.box_width
            y0 = (torch.rand(1).item() - 0.5) * self.box_height
            self._start_pos_tensor = torch.tensor([x0, y0], dtype=torch.float32)
        else:
            self._start_pos_tensor = None
        
        input, target = self._simulate_trajectories(device="cpu")
        full_dataset = TensorDataset(input, target)

        # split into train and val
        train_size = int(self.train_val_split * len(full_dataset))
        val_size = len(full_dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(
            full_dataset, [train_size, val_size]
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
            shuffle=False,
        )