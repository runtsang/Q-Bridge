import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class QLSTMGen(nn.Module):
    """
    Hybrid LSTM where a small variational quantum circuit controls the forget
    gate while the remaining gates are implemented classically.  The
    forget gate learns an entangled 𝐑𝑒𝑙 𝑒𝑒 ‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑ 
    state.  The design allows a user to toggle the quantum component by
    passing ``n_qubits=0``.
    """

    class _QuantumForgetGate(nn.Module):
        def __init__(self, n_qubits: int, device: torch.device):
            super().__init__()
            self.n_qubits = n_qubits
            self.device = device
            # Simple variational layer: 3‑layer QNN with 1‑qubit gates
            self._qnn = nn.ModuleList([nn.Linear(1, 1) for _ in range(3)])
            # Parameters for quantum‑like operations (simulated)
            self._params = nn.Parameter(torch.randn(n_qubits))

        # ...
