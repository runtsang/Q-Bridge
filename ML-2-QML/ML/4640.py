import torch
from torch import nn
import numpy as np
from.quantum_module import HybridConv as QuantumHybridConv


class HybridConv(nn.Module):
    """
    Classical + quantum convolutional layer.

    Parameters
    ----------
    kernel_size : int, default 2
        Size of the square convolution kernel.
    threshold : float, default 0.0
        Binary threshold used to encode classical activations
        into the quantum circuit.
    use_quantum : bool, default True
        If True, the quantum filter is evaluated and its output
        is combined with the classical path.
    shots : int, default 100
        Number of shots for the quantum simulator.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        threshold: float = 0.0,
        use_quantum: bool = True,
        shots: int = 100,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.use_quantum = use_quantum
        self.shots = shots

        # Classical convolution: 1 input channel → 1 output channel
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)
        self.sigmoid = nn.Sigmoid()

        # Quantum submodule (optional)
        if self.use_quantum:
            self.quantum = QuantumHybridConv(
                kernel_size=kernel_size,
                threshold=threshold,
                shots=shots,
            )

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        data : torch.Tensor
            Input tensor of shape (N, 1, H, W).

        Returns
        -------
        torch.Tensor
            Scalar output per batch element.
        """
        # Classical part
        logits = self.conv(data)
        activations = self.sigmoid(logits - self.threshold)
        classical_out = activations.mean(dim=(2, 3))  # shape (N, 1)

        if not self.use_quantum:
            return classical_out.squeeze(-1)

        # Quantum part
        # Convert activations to numpy for the quantum circuit
        numpy_data = classical_out.detach().cpu().numpy().reshape(
            -1, self.kernel_size, self.kernel_size
        )
        quantum_out = []
        for sample in numpy_data:
            quantum_out.append(
                self.quantum.run(sample, device=data.device)
            )
        quantum_out = torch.tensor(
            quantum_out, dtype=data.dtype, device=data.device
        )

        # Merge classical and quantum signals
        return 0.5 * classical_out.squeeze(-1) + 0.5 * quantum_out


__all__ = ["HybridConv"]
