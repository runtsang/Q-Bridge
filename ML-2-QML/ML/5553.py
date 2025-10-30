"""
Hybrid self‑attention module with optional quantum circuit.
The class can run in classical mode (soft‑max) or quantum mode (Qiskit)
and is fully compatible with the original SelfAttention interface.
A FastEstimator wrapper is provided for shot‑noised evaluation.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence, Tuple, List, Callable, Any

import numpy as np
import torch
from torch import nn

# Quantum imports
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

# Estimator utilities
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.circuit import QuantumCircuit as QC

# Re‑use the FastBaseEstimator/Estimator from reference pair 2
class FastBaseEstimator:
    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def evaluate(
        self,
        observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = []
        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = torch.as_tensor(params, dtype=torch.float32).unsqueeze(0)
                outputs = self.model(inputs)
                row: List[float] = []
                for observable in observables:
                    value = observable(outputs)
                    if isinstance(value, torch.Tensor):
                        scalar = float(value.mean().cpu())
                    else:
                        scalar = float(value)
                    row.append(scalar)
                results.append(row)
        return results

class FastEstimator(FastBaseEstimator):
    def evaluate(
        self,
        observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        raw = super().evaluate(observables, parameter_sets)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in raw:
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy

# Quantum‑enhanced LSTM (from reference pair 3)
class QLSTM(nn.Module):
    class QLayer(nn.Module):
        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            self.encoder = nn.ModuleList(
                [nn.Linear(1, 1, bias=False) for _ in range(n_wires)]
            )
            self.params = nn.ParameterList(
                [nn.Parameter(torch.rand(1)) for _ in range(n_wires)]
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # Simplified mock‑up: use a linear layer followed by a
            # measurement‑style operation to mimic a quantum circuit.
            out = x
            for gate, param in zip(self.encoder, self.params):
                out = torch.tanh(gate(out) + param)
            return out

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.forget = self.QLayer(n_qubits)
        self.input = self.QLayer(n_qubits)
        self.update = self.QLayer(n_qubits)
        self.output = self.QLayer(n_qubits)
        self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)

    def forward(
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget(self.linear_forget(combined)))
            i = torch.sigmoid(self.input(self.linear_input(combined)))
            g = torch.tanh(self.update(self.linear_update(combined)))
            o = torch.sigmoid(self.output(self.linear_output(combined)))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        outputs = torch.cat(outputs, dim=0)
        return outputs, (hx, cx)

    def _init_states(
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (
            torch.zeros(batch_size, self.hidden_dim, device=device),
            torch.zeros(batch_size, self.hidden_dim, device=device),
        )

# Fully‑connected quantum layer (from reference pair 4)
class FullyConnectedLayer(nn.Module):
    def __init__(self, n_features: int = 1) -> None:
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        values = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
        expectation = torch.tanh(self.linear(values)).mean(dim=0)
        return expectation.detach().numpy()

# Quantum attention circuit
class QuantumAttentionCircuit:
    """A Qiskit circuit that produces a vector of attention scores."""
    def __init__(self, n_qubits: int, backend: Any = None, shots: int = 1024):
        self.n_qubits = n_qubits
        self.backend = backend or qiskit.Aer.get_backend("qasm_simulator")
        self.shots = shots

    def _build_circuit(
        self, rotation_params: np.ndarray, entangle_params: np.ndarray
    ) -> QuantumCircuit:
        qr = QuantumRegister(self.n_qubits, "q")
        cr = ClassicalRegister(self.n_qubits, "c")
        circuit = QuantumCircuit(qr, cr)
        for i in range(self.n_qubits):
            circuit.rx(rotation_params[3 * i], i)
            circuit.ry(rotation_params[3 * i + 1], i)
            circuit.rz(rotation_params[3 * i + 2], i)
        for i in range(self.n_qubits - 1):
            circuit.crx(entangle_params[i], i, i + 1)
        circuit.measure(qr, cr)
        return circuit

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
    ) -> np.ndarray:
        circuit = self._build_circuit(rotation_params, entangle_params)
        job = qiskit.execute(circuit, self.backend, shots=self.shots)
        result = job.result()
        counts = result.get_counts(circuit)
        probs = np.array(list(counts.values())) / self.shots
        states = np.array([int(k, 2) for k in counts.keys()]).astype(float)
        expectation = np.sum(states * probs)
        # Return a vector of size n_qubits (here we just replicate the scalar)
        return np.full(self.n_qubits, expectation)

# Hybrid self‑attention module
class HybridSelfAttention(nn.Module):
    """Combines classical soft‑max attention with an optional quantum circuit."""
    def __init__(self, embed_dim: int, n_qubits: int = 0, quantum_shots: int = 1024):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_qubits = n_qubits
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        if n_qubits > 0:
            self.quantum = QuantumAttentionCircuit(n_qubits, shots=quantum_shots)
        else:
            self.quantum = None
        self.fcl = FullyConnectedLayer(n_features=embed_dim)

    def forward(
        self,
        inputs: torch.Tensor,
        rotation_params: np.ndarray | None = None,
        entangle_params: np.ndarray | None = None,
    ) -> torch.Tensor:
        """Compute attention output.  If quantum is enabled, rotation_params and
        entangle_params must be provided; otherwise the inputs are ignored."""
        q = self.query(inputs)
        k = self.key(inputs)
        v = self.value(inputs)
        if self.quantum is not None and rotation_params is not None and entangle_params is not None:
            # Quantum attention produces a score vector; broadcast to batch
            scores_np = self.quantum.run(rotation_params, entangle_params)
            scores = torch.as_tensor(scores_np, dtype=torch.float32, device=inputs.device)
            scores = scores.unsqueeze(0).expand(inputs.size(0), -1)
        else:
            scores = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(self.embed_dim), dim=-1)
        out = torch.matmul(scores, v)
        return out

    def evaluate(
        self,
        observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """Wrap FastEstimator to provide shot‑noised evaluation."""
        estimator = FastEstimator(self, shots=shots, seed=seed) if shots else FastBaseEstimator(self)
        return estimator.evaluate(observables, parameter_sets)

# Hybrid tagger that can use the quantum LSTM
class HybridTagger(nn.Module):
    """Sequence tagger that switches to a quantum LSTM when n_qubits > 0."""
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
    ) -> None:
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits > 0:
            self.lstm = QLSTM(embedding_dim, hidden_dim, n_qubits=n_qubits)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return torch.log_softmax(tag_logits, dim=1)

# Factory functions matching the original API
def SelfAttention() -> HybridSelfAttention:
    """Return a HybridSelfAttention instance with a default 4‑dim embed."""
    return HybridSelfAttention(embed_dim=4)

def Tagger() -> HybridTagger:
    """Return a HybridTagger instance configured for a toy vocabulary."""
    return HybridTagger(
        embedding_dim=50,
        hidden_dim=64,
        vocab_size=10000,
        tagset_size=10,
        n_qubits=0,
    )

__all__ = ["HybridSelfAttention", "HybridTagger", "SelfAttention", "Tagger"]
