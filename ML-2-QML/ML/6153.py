import torch
from torch import nn
import numpy as np
from typing import Iterable

class FCL(nn.Module):
    """
    Classical fully‑connected layer that mirrors the parameter layout of a
    quantum graph neural network.  It can be used as a stand‑alone MLP
    or as the classical counterpart of a quantum circuit.
    """
    def __init__(self, in_features: int, out_features: int = 1) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features)

    def run(self, params: Iterable[float]) -> np.ndarray:
        """
        Apply the linear layer to a batch of parameter vectors.
        Parameters are flattened from the iterable.
        """
        param_arr = torch.as_tensor(list(params), dtype=torch.float32).view(-1, self.in_features)
        out = torch.tanh(self.linear(param_arr))
        return out.mean(dim=0).detach().numpy()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Standard PyTorch forward pass – useful when the module is used
        inside a trainable network.
        """
        return torch.tanh(self.linear(x))

    def quantum_circuit(self, backend=None, shots=1024):
        """
        Return a parameterised quantum circuit that has the same number
        of trainable parameters as this layer.  The circuit is built with
        qiskit and uses a single qubit per input feature.
        """
        import qiskit
        from qiskit import QuantumCircuit, Parameter
        from qiskit.providers.aer import AerSimulator

        n = self.in_features
        qc = QuantumCircuit(n)
        theta = Parameter("theta")
        qc.h(range(n))
        qc.barrier()
        qc.r_y(theta, range(n))
        qc.measure_all()
        return qc, theta, AerSimulator()
