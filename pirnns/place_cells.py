import numpy as np
import scipy
import torch


class PlaceCells:
    def __init__(self, config, us=None):
        self.Np = config["output_size"]  # Number of place cells == model output size
        self.sigma = config["place_cell_rf"]
        self.surround_scale = config["surround_scale"]
        self.box_width = float(config["box_width"])
        self.box_height = float(config["box_height"])
        self.is_periodic = bool(config["periodic"])
        self.DoG = bool(config["DoG"])
        self.device = config["device"]

        self.softmax = torch.nn.Softmax(dim=-1)


        # Place-cell centers (tile uniformly across the centered box)
        if us is None:
            rng = np.random.default_rng(seed=0)
            usx = rng.uniform(-self.box_width / 2, self.box_width / 2, (self.Np,))
            usy = rng.uniform(-self.box_height / 2, self.box_height / 2, (self.Np,))
            us = np.vstack([usx, usy]).T

        self.us = torch.tensor(us, dtype=torch.float32, device=self.device)


    def get_activation(self, pos: torch.Tensor) -> torch.Tensor:
        """
        Compute place-cell activations for positions.

        Args:
            pos: (batch, T, 2) positions in the centered box.

        Returns:
            (batch, T, Np) activations (softmax-normalized; DoG-adjusted if enabled).
        """
        # Distances to each place-cell center
        # pos: (B,T,2), us: (Np,2) -> broadcast to (B,T,Np,2)
        d = torch.abs(pos[:, :, None, :] - self.us[None, None, ...]).float()

        # Periodic wrap if enabled
        if self.is_periodic:
            dx = d[..., 0]
            dy = d[..., 1]
            dx = torch.minimum(dx, self.box_width - dx)
            dy = torch.minimum(dy, self.box_height - dy)
            d = torch.stack([dx, dy], dim=-1)

        norm2 = (d ** 2).sum(-1)  # (B,T,Np)

        # Center Gaussian
        outputs = self.softmax(-norm2 / (2 * self.sigma ** 2))

        if self.DoG:
            # Surround Gaussian
            surround = self.softmax(-norm2 / (2 * self.surround_scale * self.sigma ** 2))
            outputs = outputs - surround

            # Shift to be non-negative and renormalize over Np
            min_output, _ = outputs.min(-1, keepdims=True)
            outputs = outputs + torch.abs(min_output)
            outputs = outputs / outputs.sum(-1, keepdims=True).clamp_min(1e-12)

        return outputs

    def get_nearest_cell_pos(self, activation: torch.Tensor, k: int = 3) -> torch.Tensor:
        """
        Decode position as mean of k most active place-cell centers.

        Args:
            activation: (batch, T, Np)
            k: top-k to average

        Returns:
            (batch, T, 2) decoded positions
        """
        _, idxs = torch.topk(activation, k=k, dim=-1)  # (B,T,k)
        return self.us[idxs].mean(-2)  # (B,T,2)

    def grid_pc(self, pc_outputs: np.ndarray, res: int = 32) -> np.ndarray:
        """
        Interpolate place-cell outputs onto a spatial grid.

        Args:
            pc_outputs: (..., Np) activations (will be flattened to (T, Np))
            res: grid resolution per axis

        Returns:
            (T, res, res) array
        """
        coordsx = np.linspace(-self.box_width / 2, self.box_width / 2, res)
        coordsy = np.linspace(-self.box_height / 2, self.box_height / 2, res)
        grid_x, grid_y = np.meshgrid(coordsx, coordsy)
        grid = np.stack([grid_x.ravel(), grid_y.ravel()]).T  # (res*res, 2)

        pc_outputs = pc_outputs.reshape(-1, self.Np)
        T = pc_outputs.shape[0]
        pc = np.zeros([T, res, res], dtype=np.float32)

        us_cpu = self.us.detach().cpu().numpy()
        for i in range(T):
            gridval = scipy.interpolate.griddata(us_cpu, pc_outputs[i], grid)
            pc[i] = gridval.reshape([res, res])
        return pc

    def compute_covariance(self, res: int = 30) -> np.ndarray:
        """
        Compute spatial covariance matrix over the grid of positions.

        Returns:
            (res, res) covariance image (numpy)
        """
        # Build grid of positions in the centered box
        pos = np.array(
            np.meshgrid(
                np.linspace(-self.box_width / 2, self.box_width / 2, res),
                np.linspace(-self.box_height / 2, self.box_height / 2, res),
            )
        ).T  # shape (res, res, 2)
        pos = torch.tensor(pos, dtype=torch.float32, device=self.device)

        # Activations on the grid
        pc_outputs = self.get_activation(pos).reshape(-1, self.Np)  # (res*res, Np)

        # Covariance (Gram) matrix on the grid
        C = pc_outputs @ pc_outputs.T  # (res*res, res*res)
        Csquare = C.view(res, res, res, res)

        # Average over all relative offsets
        Cmean = torch.zeros((res, res), dtype=pc_outputs.dtype, device=pc_outputs.device)
        for i in range(res):
            for j in range(res):
                Cmean += torch.roll(torch.roll(Csquare[i, j], -i, dims=0), -j, dims=1)

        Cmean = torch.roll(torch.roll(Cmean, res // 2, dims=0), res // 2, dims=1)

        return Cmean.detach().cpu().numpy()
