import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf


class HybridSamplerQNN(tq.QuantumModule):
    """
    Quantum sampler that receives classical parameters produced by the
    HybridSamplerQNN head. The circuit is a direct TorchQuantum
    implementation of the Qiskit SamplerQNN, with 2 input rotations,
    an entangling CX, and 4 weight rotations.
    """

    def __init__(self, n_wires: int = 2):
        super().__init__()
        self.n_wires = n_wires
        # Default trainable parameters (can be overwritten by forward)
        self.input_params = nn.Parameter(torch.randn(2))
        self.weight_params = nn.Parameter(torch.randn(4))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(
        self,
        x: torch.Tensor = None,
        input_params: torch.Tensor | None = None,
        weight_params: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Execute the variational sampler.

        Args:
            x: Optional placeholder for batch size. If None, batch size is 1.
            input_params: Tensor of shape [batch, 2] or [2].
            weight_params: Tensor of shape [batch, 4] or [4].

        Returns:
            Probability distribution over 2^n_wires outcomes.
        """
        bsz = x.shape[0] if x is not None else 1

        # Resolve parameters
        if input_params is None:
            input_params = self.input_params.repeat(bsz, 1)
        if weight_params is None:
            weight_params = self.weight_params.repeat(bsz, 1)

        # Quantum device
        qdev = tq.QuantumDevice(
            n_wires=self.n_wires,
            bsz=bsz,
            device=x.device if x is not None else torch.device("cpu"),
            record_op=True,
        )

        # Input rotations
        tqf.ry(qdev, input_params[:, 0], wires=0)
        tqf.ry(qdev, input_params[:, 1], wires=1)

        # Entanglement
        tqf.cx(qdev, wires=[0, 1])

        # Weight rotations
        tqf.ry(qdev, weight_params[:, 0], wires=0)
        tqf.ry(qdev, weight_params[:, 1], wires=1)
        tqf.cx(qdev, wires=[0, 1])
        tqf.ry(qdev, weight_params[:, 2], wires=0)
        tqf.ry(qdev, weight_params[:, 3], wires=1)

        # Measurement
        out = self.measure(qdev)
        probs = out / out.sum(-1, keepdim=True)
        return probs


__all__ = ["HybridSamplerQNN"]
