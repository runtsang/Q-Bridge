"""Combined classical–quantum estimator that extends the original FastBaseEstimator
by allowing the user to attach a quantum self‑attention block (Qiskit) and/or a
quantum LSTM cell (TorchQuantum).  The estimator can operate deterministically,
add Gaussian shot noise, or forward a sequence through the chosen quantum
sub‑modules, yielding a flexible hybrid workflow."""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional, Tuple
import numpy as np
import torch
from torch import nn

# Try to import the original base estimator; fall back to a minimal stub
try:
    from.FastBaseEstimator import FastBaseEstimator, FastEstimator  # type: ignore
except Exception:  # pragma: no cover
    class FastBaseEstimator:
        def __init__(self, model: nn.Module) -> None:
            self.model = model

        def evaluate(
            self,
            observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]],
            parameter_sets: Sequence[Sequence[float]],
        ) -> List[List[float]]:
            raise NotImplementedError

    class FastEstimator(FastBaseEstimator):
        pass

# Optional quantum sub‑modules – imported lazily to keep the ML module free of
# heavy quantum dependencies when not required.
try:
    from.quantum_modules import QuantumSelfAttention, QuantumQLSTM
except Exception:  # pragma: no cover
    QuantumSelfAttention = None  # type: ignore
    QuantumQLSTM = None  # type: ignore


class FastBaseEstimatorGen128(FastBaseEstimator):
    """A hybrid estimator that can delegate attention and LSTM gating to
    quantum backends while preserving the deterministic evaluation pipeline
    of the original FastBaseEstimator."""
    def __init__(
        self,
        model: nn.Module,
        *,
        use_q_attention: bool = False,
        use_q_lstm: bool = False,
        shots: int | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__(model)
        self.use_q_attention = use_q_attention
        self.use_q_lstm = use_q_lstm
        self.shots = shots
        self.seed = seed

        # Placeholders for quantum modules; instantiated on demand
        self.attention: Optional[QuantumSelfAttention] = None
        self.lstm: Optional[QuantumQLSTM] = None

        if self.use_q_attention and QuantumSelfAttention is not None:
            self.attention = QuantumSelfAttention()
        if self.use_q_lstm and QuantumQLSTM is not None:
            self.lstm = QuantumQLSTM()

    # --------------------------------------------------------------------- #
    #  Quantum sub‑module helpers
    # --------------------------------------------------------------------- #
    def set_attention(self, attention: QuantumSelfAttention) -> None:
        """Attach a custom quantum self‑attention module."""
        self.attention = attention
        self.use_q_attention = True

    def set_lstm(self, lstm: QuantumQLSTM) -> None:
        """Attach a custom quantum LSTM module."""
        self.lstm = lstm
        self.use_q_lstm = True

    # --------------------------------------------------------------------- #
    #  Core evaluation logic
    # --------------------------------------------------------------------- #
    def evaluate(
        self,
        observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        # 1. Deterministic forward pass using the underlying PyTorch model
        raw_results = super().evaluate(observables, parameter_sets)

        # 2. Optional quantum attention post‑processing
        if self.use_q_attention and self.attention is not None:
            new_results: List[List[float]] = []
            for row in raw_results:
                # Treat each row as a feature vector for the attention block
                inputs = np.array(row, dtype=np.float64)
                # Generate dummy parameters – in practice these would be learned
                rot_params = np.random.uniform(0, 2 * np.pi,
                                               size=self.attention.n_qubits * 3)
                ent_params = np.random.uniform(0, 2 * np.pi,
                                               size=self.attention.n_qubits - 1)
                counts = self.attention.run(
                    backend=self.attention.backend,
                    rotation_params=rot_params,
                    entangle_params=ent_params,
                    shots=self.shots or 1024,
                )
                # Convert measurement counts to a single scalar expectation value
                total_shots = sum(counts.values())
                exp_val = sum(int(k) * v for k, v in counts.items()) / total_shots
                # Replicate the scalar across the row length to preserve shape
                new_results.append([exp_val] * len(row))
            raw_results = new_results

        # 3. Optional quantum LSTM post‑processing – placeholder logic
        if self.use_q_lstm and self.lstm is not None:
            # In a realistic scenario, the LSTM would be integrated into the
            # model architecture; here we simply demonstrate a dummy pass.
            pass  # pragma: no cover

        # 4. Optional shot‑noise injection (classical Gaussian noise)
        if self.shots is not None:
            rng = np.random.default_rng(self.seed)
            noisy_results: List[List[float]] = []
            for row in raw_results:
                noise_std = max(1e-6, 1 / self.shots)
                noisy_row = [float(rng.normal(mean, noise_std)) for mean in row]
                noisy_results.append(noisy_row)
            raw_results = noisy_results

        return raw_results

    # --------------------------------------------------------------------- #
    #  Convenience methods for sequence handling (self‑attention + LSTM)
    # --------------------------------------------------------------------- #
    def run_sequence(
        self,
        sequence: torch.Tensor,
        *,
        attention_params: Tuple[np.ndarray, np.ndarray] | None = None,
        lstm_init: Tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Run a sequence through the underlying model, optionally applying
        quantum self‑attention and LSTM gating."""
        # Forward through the PyTorch model (assumed to be a sequence model)
        outputs, _ = self.model(sequence)
        # Apply quantum self‑attention if requested
        if self.use_q_attention and self.attention is not None and attention_params is not None:
            rot, ent = attention_params
            counts = self.attention.run(
                backend=self.attention.backend,
                rotation_params=rot,
                entangle_params=ent,
                shots=self.shots or 1024,
            )
            # Convert counts to a tensor; here we simply use the mean of bitstring keys
            total_shots = sum(counts.values())
            exp_val = sum(int(k) * v for k, v in counts.items()) / total_shots
            attention_tensor = torch.full_like(outputs, exp_val)
            outputs = outputs + attention_tensor
        # Apply quantum LSTM if requested – placeholder
        if self.use_q_lstm and self.lstm is not None:
            # In a real implementation, the LSTM would be a module; here we skip.
            pass  # pragma: no cover
        return outputs

__all__ = ["FastBaseEstimatorGen128"]
