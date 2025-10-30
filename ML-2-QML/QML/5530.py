import torch
import torchquantum as tq
import torchquantum.functional as tqf
import numpy as np

class QuantumKernelMethod(tq.QuantumModule):
    """Quantum kernel that computes the squared overlap between
    parameterised quantum states encoded from two input batches.
    The encoder uses a trainable RX rotation per feature followed by
    a fixed entangling layer and a random layer to enrich the feature
    space.  The kernel matrix is returned as a NumPy array for
    compatibility with classical pipelines.
    """

    def __init__(self,
                 input_dim: int,
                 n_wires: int = None,
                 random_ops: int = 50,
                 seed: int = 42):
        super().__init__()
        self.input_dim = input_dim
        self.n_wires = n_wires or input_dim
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.random_layer = tq.RandomLayer(n_ops=random_ops,
                                          wires=list(range(self.n_wires)),
                                          seed=seed)

    def _encode(self, qdev: tq.QuantumDevice, data: torch.Tensor) -> torch.Tensor:
        """
        Encode a batch of feature vectors into the quantum device.
        ``data`` is of shape ``(batch, input_dim)``.
        Returns the state‑vector of shape ``(batch, 2**n_wires)``.
        """
        qdev.reset_states(data.size(0))
        for wire in range(self.n_wires):
            # Use RX rotation for each feature; if input_dim < n_wires pad with zeros
            param = data[:, wire] if wire < data.size(1) else torch.zeros(data.size(0), device=data.device)
            qdev.rx(wires=wire, params=param)
        self.random_layer(qdev)
        # Entangle with a ring of CNOTs
        for wire in range(self.n_wires - 1):
            tqf.cnot(qdev, wires=[wire, wire + 1])
        tqf.cnot(qdev, wires=[self.n_wires - 1, 0])
        return qdev.state_vector

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the kernel matrix between two batches of inputs.
        ``x`` and ``y`` must have shape ``(batch, input_dim)``.
        Returns a tensor of shape ``(len(x), len(y))``.
        """
        sv_x = self._encode(self.q_device, x)
        sv_y = self._encode(self.q_device, y)
        # Compute squared overlap |<ψ_x|ψ_y>|^2
        kernel = torch.einsum('bi,bj->ij', sv_x, sv_y.conj()).abs() ** 2
        return kernel

    def kernel_matrix(self, a: torch.Tensor, b: torch.Tensor) -> np.ndarray:
        """Return the kernel matrix as a NumPy array."""
        return self.forward(a, b).detach().cpu().numpy()

__all__ = ["QuantumKernelMethod"]
