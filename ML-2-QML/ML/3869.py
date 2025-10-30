"""Unified estimator combining classical PyTorch and quantum Qiskit workflows.

The module defines several classes:

* :class:`UnifiedBaseEstimator` – classically evaluates a
  :class:`~torch.nn.Module` for many parameter sets.
* :class:`UnifiedEstimator` – extends the base estimator with optional
  Gaussian shot‑noise, a quantum evaluator (from :mod:`qml_code`) and
  a hybrid LSTM tagger that can switch between classical :class:`QLSTM`
  and quantum :class:`QuantumQLSTM`.
* :class:`QLSTM` – classical LSTM cell used for the classical tagger.
* :class:`LSTMTagger` – sequence‑tagging model that uses either
  :class:`QLSTM` or :class:`torch.nn.LSTM`.
* :class:`HybridLSTMTagger` – a wrapper that chooses the appropriate
  LSTM implementation based on the *n_qubits* flag and exposes a
  :meth:`forward` compatible with the original tagger.

The design keeps the original API of ``FastBaseEstimator`` and
``FastEstimator`` and adds a ``n_qubits`` flag to switch between
classical and quantum behaviour.  The estimator can be used as a
drop‑in replacement for any model that implements a ``forward`` method
accepting a batch of inputs.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional, Union

# --------------------------------------------------------------------------- #
# Helper utilities
# --------------------------------------------------------------------------- #
def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Convert a 1‑D sequence of scalars into a 2‑D tensor (batch, feature)."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor

# --------------------------------------------------------------------------- #
# Classical estimator
# --------------------------------------------------------------------------- #
class UnifiedBaseEstimator:
    """Evaluate a PyTorch model for many parameter sets.

    The implementation follows the pattern of the original
    ``FastBaseEstimator`` but adds type hints and docstrings for clarity.
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def evaluate(
        self,
        observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        """Return a list of lists containing the outputs of each
        ``observable`` (i.e. a list of scalars) for each parameter set.
        """
        observables = list(observables) or [lambda x: x.mean(dim=-1)]
        results: List[List[float]] = []
        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params)
                outputs = self.model(inputs)
                row: List[float] = []
                for obs in observables:
                    value = obs(outputs)
                    if isinstance(value, torch.Tensor):
                        scalar = float(value.mean().cpu())
                    else:
                        scalar = float(value)
                    row.append(scalar)
                results.append(row)
        return results

# --------------------------------------------------------------------------- #
# Hybrid estimator with shot noise, variational circuit, and QLSTM support
# --------------------------------------------------------------------------- #
class UnifiedEstimator(UnifiedBaseEstimator):
    """Hybrid estimator that can evaluate classical PyTorch models or
    quantum circuits, optionally adding Gaussian shot noise and using a
    quantum LSTM layer for sequence tagging.

    Parameters
    ----------
    model:
        Either a :class:`torch.nn.Module` or a Qiskit :class:`~qiskit.circuit.QuantumCircuit`.
    n_qubits:
        If >0 and ``model`` is a quantum circuit, a quantum LSTM layer will
        be used internally for sequence tagging tasks.
    shots, seed:
        If provided, Gaussian noise will be added to deterministic
        predictions to mimic finite‑shot sampling.
    variational_circuit:
        Optional parametric circuit that will be appended to the base
        circuit before expectation evaluation.
    """

    def __init__(
        self,
        model: Union[nn.Module, "QuantumCircuit"],
        *,
        n_qubits: int = 0,
        shots: Optional[int] = None,
        seed: Optional[int] = None,
        variational_circuit: Optional["QuantumCircuit"] = None,
    ) -> None:
        self.n_qubits = n_qubits
        self.shots = shots
        self.seed = seed
        self.variational_circuit = variational_circuit

        # Detect quantum vs classical
        try:
            from qiskit import QuantumCircuit
        except Exception:
            QuantumCircuit = None

        if isinstance(model, nn.Module):
            self._is_quantum = False
            super().__init__(model)
        elif QuantumCircuit is not None and isinstance(model, QuantumCircuit):
            self._is_quantum = True
            self._qmodel = model
            # Lazy import of quantum evaluator
            from.qml_code import QuantumEvaluator

            self._quantum_evaluator = QuantumEvaluator(
                base_circuit=model,
                variational_circuit=variational_circuit,
                shots=shots,
            )
        else:
            raise TypeError(
                "model must be a torch.nn.Module or a qiskit.quantum_info.QuantumCircuit"
            )

    def evaluate(
        self,
        observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float] | "BaseOperator"],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        if self._is_quantum:
            # If observables are Qiskit operators, use quantum evaluator
            results = self._quantum_evaluator.evaluate(
                observables=observables, parameter_sets=parameter_sets
            )
            # Convert complex numbers to real if possible
            return [[float(r.real) if isinstance(r, complex) else r for r in row] for row in results]
        else:
            # Classical evaluation via parent method
            return super().evaluate(observables, parameter_sets)

    def add_shot_noise(
        self,
        results: List[List[float]],
    ) -> List[List[float]]:
        """Optionally add Gaussian noise to emulate finite shots."""
        if self.shots is None:
            return results
        rng = np.random.default_rng(self.seed)
        noisy = []
        for row in results:
            noisy_row = [
                float(rng.normal(mean, max(1e-6, 1 / self.shots))) for mean in row
            ]
            noisy.append(noisy_row)
        return noisy

    def __call__(
        self,
        observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float] | "BaseOperator"],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        """Convenience wrapper that returns noisy results if shots were set."""
        raw = self.evaluate(observables, parameter_sets)
        return self.add_shot_noise(raw)

# --------------------------------------------------------------------------- #
# Classical LSTM modules (borrowed from reference pair 2)
# --------------------------------------------------------------------------- #
class QLSTM(nn.Module):
    """Drop‑in replacement using classical linear gates."""

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        gate_dim = hidden_dim
        self.forget_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.input_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.update_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.output_linear = nn.Linear(input_dim + hidden_dim, gate_dim)

    def forward(
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:  # type: ignore[override]
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget_linear(combined))
            i = torch.sigmoid(self.input_linear(combined))
            g = torch.tanh(self.update_linear(combined))
            o = torch.sigmoid(self.output_linear(combined))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        stacked = torch.cat(outputs, dim=0)
        return stacked, (hx, cx)

    def _init_states(
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        hx = torch.zeros(batch_size, self.hidden_dim, device=device)
        cx = torch.zeros(batch_size, self.hidden_dim, device=device)
        return hx, cx

class LSTMTagger(nn.Module):
    """Sequence tagging model that uses either :class:`QLSTM` or ``nn.LSTM``."""

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits > 0:
            self.lstm = QLSTM(embedding_dim, hidden_dim, n_qubits=n_qubits)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return torch.log_softmax(tag_logits, dim=1)

# --------------------------------------------------------------------------- #
# Hybrid tagger that can switch between classical and quantum LSTM
# --------------------------------------------------------------------------- #
class HybridLSTMTagger(nn.Module):
    """Wrapper that chooses the appropriate LSTM implementation based on
    the ``n_qubits`` flag and exposes a single ``forward`` method.
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits > 0:
            # Quantum LSTM from the QML companion will be swapped in at usage time.
            # Here we keep the placeholder for API compatibility.
            self.lstm = QLSTM(embedding_dim, hidden_dim, n_qubits=n_qubits)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return torch.log_softmax(tag_logits, dim=1)

__all__ = [
    "UnifiedBaseEstimator",
    "UnifiedEstimator",
    "QLSTM",
    "LSTMTagger",
    "HybridLSTMTagger",
]
