"""Combined classical estimator that supports both feed‑forward regression and RBF‑kernel feature maps.

The class exposes a unified interface while allowing the user to switch between a pure neural network
and a kernelised linear model.  The kernel centres can be learned from data or supplied
externally, enabling a smooth transition between classical and quantum‑inspired feature
engineering.
"""

from __future__ import annotations

import torch
from torch import nn
from typing import Optional, Sequence, Tuple


class CombinedEstimatorQNN(nn.Module):
    """
    A hybrid classical estimator that can operate in two modes:

    1. **Feed‑forward mode** – a small multi‑layer perceptron identical to the original
       EstimatorQNN implementation.
    2. **Kernel mode** – the RBF kernel from *QuantumKernelMethod* is used as a feature map,
       followed by a single linear layer.  The kernel centres are either fixed or learned
       via an auxiliary network.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input vectors.
    hidden_dim : int, default 8
        Number of units in the hidden layers (feed‑forward mode only).
    output_dim : int, default 1
        Output dimensionality.
    gamma : float, default 1.0
        Width parameter of the RBF kernel.
    use_kernel : bool, default False
        If ``True`` the model operates in kernel mode.
    kernel_centers : Sequence[torch.Tensor] | None, default None
        Optional pre‑computed centres for the RBF kernel.  If ``None`` the centres will
        be learned during the first ``fit`` call via a simple linear embedding.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 8,
        output_dim: int = 1,
        gamma: float = 1.0,
        use_kernel: bool = False,
        kernel_centers: Optional[Sequence[torch.Tensor]] = None,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.use_kernel = use_kernel
        self.gamma = gamma

        if self.use_kernel:
            if kernel_centers is not None:
                centers = torch.stack(kernel_centers, dim=0)
            else:
                # initialise centres with zeros; will be updated in ``fit_kernel_centers``
                centers = torch.empty(0, input_dim, dtype=torch.float32)
            self.register_buffer("kernel_centers", centers)
            self.linear = nn.Linear(self.kernel_centers.shape[0], output_dim)
        else:
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 4),
                nn.Tanh(),
                nn.Linear(4, output_dim),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(batch, input_dim)``.

        Returns
        -------
        torch.Tensor
            Predicted outputs, shape ``(batch, output_dim)``.
        """
        if self.use_kernel:
            if self.kernel_centers.shape[0] == 0:
                raise RuntimeError(
                    "Kernel centres are not set. Call ``fit_kernel_centers`` before prediction."
                )
            # Compute RBF kernel between inputs and centres
            diff = x.unsqueeze(1) - self.kernel_centers.unsqueeze(0)  # (batch, n_centres, dim)
            dist_sq = (diff**2).sum(-1)  # (batch, n_centres)
            k = torch.exp(-self.gamma * dist_sq)  # (batch, n_centres)
            return self.linear(k)
        else:
            return self.net(x)

    # ----------------------------------------------------------------------
    # Utility helpers
    # ----------------------------------------------------------------------
    def fit_kernel_centers(self, data: torch.Tensor, n_centres: int = 50, epochs: int = 100, lr: float = 0.01) -> None:
        """
        Learn kernel centres from data using a simple linear embedding.

        Parameters
        ----------
        data : torch.Tensor
            Training data of shape ``(N, input_dim)``.
        n_centres : int, default 50
            Number of kernel centres to learn.
        epochs : int, default 100
            Training epochs.
        lr : float, default 0.01
            Learning rate.
        """
        if not self.use_kernel:
            raise RuntimeError("Kernel centres can only be fitted in kernel mode.")
        # Initialise centres as a learnable parameter
        centres = nn.Parameter(data[torch.randperm(data.shape[0])[:n_centres]])
        opt = torch.optim.Adam([centres], lr=lr)
        loss_fn = nn.MSELoss()
        for _ in range(epochs):
            k = torch.exp(-self.gamma * ((data.unsqueeze(1) - centres.unsqueeze(0)) ** 2).sum(-1))
            preds = self.linear(k)
            loss = loss_fn(preds, torch.zeros_like(preds))
            opt.zero_grad()
            loss.backward()
            opt.step()
        self.register_buffer("kernel_centers", centres.detach())
        self.linear = nn.Linear(self.kernel_centers.shape[0], self.linear.out_features)

    def set_kernel_centers(self, centres: Sequence[torch.Tensor]) -> None:
        """
        Manually set the kernel centres.

        Parameters
        ----------
        centres : Sequence[torch.Tensor]
            List or array of centre tensors, each of shape ``(input_dim,)``.
        """
        self.register_buffer("kernel_centers", torch.stack(centres, dim=0))

__all__ = ["CombinedEstimatorQNN"]
