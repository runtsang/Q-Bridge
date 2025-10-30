"""Quantum implementation of the sampler circuit used by SamplerQNNGen422.

The implementation is deliberately lightweight: a RealAmplitudes ansatz
parameterised by a small number of trainable angles, with input data
encoded via Ry rotations.  The circuit is wrapped in a callable
class that accepts a torch.Tensor of latent vectors and returns the
probability distribution over the 2‑qubit measurement outcomes.
"""

from __future__ import annotations

import numpy as np
import torch
from qiskit.circuit import ParameterVector
from qiskit import QuantumCircuit
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import StatevectorSampler
from qiskit_aer import AerSimulator
from typing import Iterable

# ---------- Quantum sampler ----------
class QuantumSampler:
    """Callable wrapper that evaluates a small quantum circuit."""

    def __init__(self, num_qubits: int = 2, reps: int = 3) -> None:
        # Input parameters are real numbers that will be applied via Ry
        self.input_params = ParameterVector("x", length=num_qubits)
        # Weight parameters are the trainable angles of the ansatz
        self.weight_params = ParameterVector("w", length=reps * num_qubits)

        # Build the template circuit
        qc = QuantumCircuit(num_qubits)
        for i in range(num_qubits):
            qc.ry(self.input_params[i], i)
        # RealAmplitudes ansatz
        qc.compose(RealAmplitudes(num_qubits, reps=reps), inplace=True)
        # Final measurement in computational basis
        qc.measure_all()

        # Store
        self.circuit = qc
        self.simulator = AerSimulator(method="statevector")
        self.state_sampler = StatevectorSampler(self.simulator)

    def __call__(self, latent: torch.Tensor) -> torch.Tensor:
        """Evaluate the circuit for a batch of latent vectors.

        Parameters
        ----------
        latent : torch.Tensor
            Shape (batch, num_qubits) where each row contains the input
            angles for the Ry gates.

        Returns
        -------
        torch.Tensor
            Probability distribution of shape (batch, 2**num_qubits).
        """
        if latent.ndim!= 2 or latent.shape[1]!= self.circuit.num_qubits:
            raise ValueError(
                f"latent must be of shape (batch, {self.circuit.num_qubits})"
            )
        # Convert to numpy for qiskit
        latent_np = latent.detach().cpu().numpy()
        probs = []
        for row in latent_np:
            # Bind parameters
            bound_circuit = self.circuit.bind_parameters(
                dict(zip(self.input_params, row))
            )
            # Sample probabilities
            result = self.state_sampler.run(bound_circuit).result()
            probs.append(result.get_counts(bound_circuit))
        # Convert counts to probability array
        probs_np = np.zeros((latent.shape[0], 2**self.circuit.num_qubits))
        for idx, count_dict in enumerate(probs):
            for state, cnt in count_dict.items():
                probs_np[idx, int(state, 2)] = cnt / sum(count_dict.values())
        return torch.from_numpy(probs_np).to(latent.device)

# ---------- Quantum kernel ----------
# Reuse TorchQuantum implementation from reference 3
import torchquantum as tq
from torchquantum.functional import func_name_dict

class KernalAnsatz(tq.QuantumModule):
    """Encodes classical data through a programmable list of quantum gates."""
    def __init__(self, func_list):
        super().__init__()
        self.func_list = func_list

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        q_device.reset_states(x.shape[0])
        for info in self.func_list:
            params = x[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)
        for info in reversed(self.func_list):
            params = -y[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)

class Kernel(tq.QuantumModule):
    """Quantum kernel evaluated via a fixed TorchQuantum ansatz."""
    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.ansatz = KernalAnsatz(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.ansatz(self.q_device, x, y)
        return torch.abs(self.q_device.states.view(-1)[0])

def kernel_matrix(a: Iterable[torch.Tensor], b: Iterable[torch.Tensor]) -> np.ndarray:
    kernel = Kernel()
    return np.array([[kernel(x, y).item() for y in b] for x in a])

__all__ = [
    "QuantumSampler",
    "KernalAnsatz",
    "Kernel",
    "kernel_matrix",
]
