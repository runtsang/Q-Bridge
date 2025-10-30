"""Quantum‑centric estimator and helper modules.

The estimator is compatible with the classical interface but adds
support for Qiskit circuits and simple quantum layers.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Tuple, Union, Any

import numpy as np
import torch
from torch import nn
import qiskit
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import Parameter
from qiskit.quantum_info import Qobj

# ----------------------------------------------------------------------
# Utility: batch conversion
# ----------------------------------------------------------------------
def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


# ----------------------------------------------------------------------
# Helper: fully connected quantum circuit (FCL)
# ----------------------------------------------------------------------
class QuantumFCL:
    """Parameterised quantum circuit that mimics a classical fully‑connected layer."""

    def __init__(self, n_qubits: int = 1, shots: int = 1000):
        self.n_qubits = n_qubits
        self.shots = shots
        self.backend = Aer.get_backend("qasm_simulator")
        self.theta = Parameter("theta")
        self.circuit = QuantumCircuit(n_qubits)
        for q in range(n_qubits):
            self.circuit.h(q)
        self.circuit.barrier()
        self.circuit.ry(self.theta, range(n_qubits))
        self.circuit.measure_all()

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        job = execute(
            self.circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=[{self.theta: theta} for theta in thetas],
        )
        result = job.result()
        counts = result.get_counts(self.circuit)
        probs = np.array(list(counts.values())) / self.shots
        states = np.array([int(s, 2) for s in counts.keys()]).astype(float)
        expectation = np.sum(states * probs)
        return np.array([expectation])


# ----------------------------------------------------------------------
# Helper: quantum LSTM layer (QLSTM)
# ----------------------------------------------------------------------
class QLSTM(nn.Module):
    """Very small quantum LSTM implementation using Qiskit parameterised gates."""

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        # Classical linear layers that produce parameters for the gates
        self.forget_linear = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.input_linear = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.update_linear = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.output_linear = nn.Linear(input_dim + hidden_dim, n_qubits)

    def _build_gate(self, params: torch.Tensor) -> QuantumCircuit:
        qc = QuantumCircuit(self.n_qubits)
        for i, p in enumerate(params):
            qc.rx(p.item(), i)
        return qc

    def forward(self, inputs: torch.Tensor, states: Tuple[torch.Tensor, torch.Tensor] | None = None):
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget_linear(combined))
            i = torch.sigmoid(self.input_linear(combined))
            g = torch.tanh(self.update_linear(combined))
            o = torch.sigmoid(self.output_linear(combined))
            # Quantum gates would be executed here; omitted for brevity
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        return torch.cat(outputs, dim=0), (hx, cx)

    def _init_states(self, inputs: torch.Tensor, states: Tuple[torch.Tensor, torch.Tensor] | None):
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (
            torch.zeros(batch_size, self.hidden_dim, device=device),
            torch.zeros(batch_size, self.hidden_dim, device=device),
        )


# ----------------------------------------------------------------------
# Helper: graph quantum neural network (GraphQNN)
# ----------------------------------------------------------------------
class GraphQNN:
    """Minimal quantum graph neural network used for fidelity experiments."""

    def __init__(self, qnn_arch: Sequence[int]):
        self.qnn_arch = list(qnn_arch)
        self.unitaries: list[list[Qobj]] = [[]]
        for layer in range(1, len(self.qnn_arch)):
            num_inputs = self.qnn_arch[layer - 1]
            num_outputs = self.qnn_arch[layer]
            layer_ops: list[Qobj] = []
            for _ in range(num_outputs):
                op = Qobj(np.eye(2 ** (num_inputs + 1)))  # placeholder
                layer_ops.append(op)
            self.unitaries.append(layer_ops)

    def feedforward(
        self, samples: Iterable[Tuple[Qobj, Qobj]]
    ) -> List[List[Qobj]]:
        stored_states: List[List[Qobj]] = []
        for sample, _ in samples:
            current = sample
            layerwise = [current]
            for layer in range(1, len(self.qnn_arch)):
                unitary = self.unitaries[layer][0]
                current = unitary @ current
                layerwise.append(current)
            stored_states.append(layerwise)
        return stored_states


# ----------------------------------------------------------------------
# Utility: fidelity
# ----------------------------------------------------------------------
def state_fidelity(a: Qobj, b: Qobj) -> float:
    return abs((a.dag() @ b)[0, 0]) ** 2


# ----------------------------------------------------------------------
# Hybrid estimator
# ----------------------------------------------------------------------
class HybridFastEstimator:
    """
    Estimator that can evaluate either a classical PyTorch model,
    a Qiskit circuit, or a hybrid object that implements a ``run`` method.
    The public API mirrors the classical FastBaseEstimator but
    includes convenience wrappers for quantum‑specific experiments.
    """

    def __init__(self, model: Any) -> None:
        self.model = model

    # Generic evaluation ----------------------------------------------------
    def evaluate(
        self,
        observables: Iterable[Callable[[Any], Any]],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """
        Evaluate a list of observables for each parameter set.

        * If ``model`` is a PyTorch module the observables must accept a
          ``torch.Tensor``.
        * If ``model`` provides a ``run`` method (e.g. Qiskit circuit)
          the observables must accept a NumPy array of expectation values.
        """
        if not observables:
            raise ValueError("At least one observable must be provided")

        results: List[List[float]] = []

        is_torch = isinstance(self.model, nn.Module)

        for params in parameter_sets:
            if is_torch:
                inputs = _ensure_batch(params)
                with torch.no_grad():
                    outputs = self.model(inputs)
                row: List[float] = []
                for obs in observables:
                    val = obs(outputs)
                    if isinstance(val, torch.Tensor):
                        val = float(val.mean().cpu())
                    else:
                        val = float(val)
                    row.append(val)
            else:
                # Assume a callable with a run method
                raw = self.model.run(params)
                if not isinstance(raw, np.ndarray):
                    raise TypeError("Quantum model must return a NumPy array")
                row = [float(obs(raw)) for obs in observables]
            results.append(row)

        if shots is not None:
            rng = np.random.default_rng(seed)
            noisy: List[List[float]] = []
            for row in results:
                noisy_row = [rng.normal(mean, max(1e-6, 1 / shots)) for mean in row]
                noisy.append(noisy_row)
            return noisy

        return results

    # Sequence evaluation ----------------------------------------------------
    def evaluate_sequence(
        self,
        sentence: torch.Tensor,
        tagset_size: int,
    ) -> torch.Tensor:
        """
        Evaluate a sequence tagging model.  The model is expected to be a
        PyTorch module that implements the same interface as the classical
        ``LSTMTagger``.
        """
        if not hasattr(self.model, "forward"):
            raise AttributeError("Model does not support sequence evaluation")
        if not isinstance(self.model, nn.Module):
            raise TypeError("Sequence evaluation requires a PyTorch module")
        embeds = self.model.word_embeddings(sentence)
        lstm_out, _ = self.model.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.model.hidden2tag(lstm_out.view(len(sentence), -1))
        return torch.log_softmax(tag_logits, dim=1)

    # Graph neural network evaluation ---------------------------------------
    def evaluate_graph(
        self,
        samples: Iterable[Tuple[Qobj, Qobj]],
    ) -> List[List[float]]:
        """
        Evaluate a graph quantum neural network.  ``samples`` should be an
        iterable of (state, target) pairs where both elements are
        Qiskit ``Qobj`` instances representing pure states.
        The method returns a list of lists containing the fidelity
        between the propagated state and the target at each layer.
        """
        if not hasattr(self.model, "feedforward"):
            raise AttributeError("Model does not provide feedforward")
        fidelities: List[List[float]] = []

        for state, target in samples:
            layerwise = self.model.feedforward([(state, target)])
            layer_fids = [float(state_fidelity(s, target)) for s in layerwise[0]]
            fidelities.append(layer_fids)
        return fidelities


__all__ = ["HybridFastEstimator", "QuantumFCL", "QLSTM", "GraphQNN"]
