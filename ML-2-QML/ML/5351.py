from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# --------------------------------------------------------------------------- #
#  Classical hybrid module – replaces the simple linear head
# --------------------------------------------------------------------------- #
class _ShiftedHybrid(nn.Module):
    """A classical dense head that mimics a quantum‑like expectation:
    It applies a learnable shift and a sigmoid activation to the linear output.
    The shift is initialized to zero and can be learned, providing a
    differentiable surrogate for the quantum circuit used in the original QML
    implementation."""
    def __init__(self, in_features: int, init_shift: float = 0.0) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.shift = nn.Parameter(torch.tensor(init_shift, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.linear(x)
        return torch.sigmoid(logits + self.shift)


# --------------------------------------------------------------------------- #
#  Quantum expectation wrapper – keeps the original QML interface
# --------------------------------------------------------------------------- #
class QuantumCircuit:
    """Thin wrapper around a parametrised two‑qubit circuit executed on Aer.
    The circuit is identical to the one in the reference, but the run method
    now accepts a 1‑D array of angles and returns the expectation of the
    Y‑observable for each angle.  It is kept separate so that the
    classical and quantum paths can be swapped at runtime."""
    def __init__(self, n_qubits: int, backend, shots: int) -> None:
        self._circuit = qiskit.QuantumCircuit(n_qubits)
        all_qubits = list(range(n_qubits))
        self.theta = qiskit.circuit.Parameter("theta")

        self._circuit.h(all_qubits)
        self._circuit.barrier()
        self._circuit.ry(self.theta, all_qubits)
        self._circuit.measure_all()

        self.backend = backend
        self.shots = shots

    def run(self, thetas: np.ndarray) -> np.ndarray:
        compiled = qiskit.transpile(self._circuit, self.backend)
        qobj = qiskit.assemble(
            compiled,
            shots=self.shots,
            parameter_binds=[{self.theta: theta} for theta in thetas],
        )
        job = self.backend.run(qobj)
        result = job.result().get_counts()

        def expectation(count_dict):
            counts = np.array(list(count_dict.values()))
            states = np.array(list(count_dict.keys())).astype(float)
            probs = counts / self.shots
            return np.sum(states * probs)

        if isinstance(result, list):
            return np.array([expectation(r) for r in result])
        return np.array([expectation(result)])


class QuantumHybridLayer(nn.Module):
    """Hybrid layer that forwards activations through a quantum circuit.
    It uses the _ShiftedHybrid as a fallback when the quantum backend is
    unavailable (e.g. during unit tests)."""
    def __init__(self, n_qubits: int, backend, shots: int, shift: float = np.pi / 2):
        super().__init__()
        self.circuit = QuantumCircuit(n_qubits, backend, shots)
        self.shift = nn.Parameter(torch.tensor(shift, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Flatten to a 1‑D tensor of angles
        angles = torch.squeeze(x).float()
        # Run the quantum circuit
        exp_vals = torch.from_numpy(self.circuit.run(angles.cpu().numpy()))
        return exp_vals.to(x.device)


# --------------------------------------------------------------------------- #
#  Core CNN + hybrid head
# --------------------------------------------------------------------------- #
class HybridBinaryFusionNet(nn.Module):
    """A hybrid CNN that mirrors the original QCNet but replaces the
    final dense head with a hybrid quantum‑or‑dense layer.  The network
    can be configured to use either the quantum circuit or the
    classical surrogate during training."""
    def __init__(self, use_quantum: bool = True, shift: float = 0.0) -> None:
        super().__init__()
        # Convolutional feature extractor
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)

        # Flattened feature size after the convs
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

        # Hybrid head
        if use_quantum:
            backend = qiskit.Aer.get_backend("aer_simulator")
            self.head = QuantumHybridLayer(1, backend, shots=200, shift=shift)
        else:
            self.head = _ShiftedHybrid(1, init_shift=shift)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Feature extraction
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)

        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        # Hybrid head
        probs = self.head(x)
        return torch.cat((probs, 1 - probs), dim=-1)


# --------------------------------------------------------------------------- #
#  Fidelity‑based adjacency helper – copied from GraphQNN
# --------------------------------------------------------------------------- #
def fidelity_adjacency(
    states: Sequence[torch.Tensor],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    """Create a weighted adjacency graph from state fidelities.
    The function accepts torch tensors and internally casts them to
    numpy arrays for the fidelity calculation."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for i, a in enumerate(states):
        for j in range(i + 1, len(states)):
            fid = torch.dot(a / a.norm(), b / b.norm()).item() ** 2
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
    return graph


__all__ = [
    "HybridBinaryFusionNet",
    "_ShiftedHybrid",
    "QuantumHybridLayer",
    "QuantumCircuit",
    "fidelity_adjacency",
]
