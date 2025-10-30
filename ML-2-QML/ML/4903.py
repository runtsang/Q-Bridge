"""
FastHybridEstimator – a unified classical/quantum estimator.

This module re‑implements the original lightweight estimator while
adding:
* Shot‑noise simulation for classical models.
* Quantum‑circuit expectation evaluation using a statevector or qasm simulator.
* Convenience constructors for quanvolution, fraud‑detection and autoencoder
  architectures, both classical and quantum.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, List, Callable, Any, Union

import numpy as np
import torch
from torch import nn
import torchquantum as tq

# Quantum imports for simulation
from qiskit import QuantumCircuit, Aer, execute
from qiskit.quantum_info import Statevector, Operator


# ----------------------------------------------------------------------
# Utility helpers
# ----------------------------------------------------------------------
def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Convert a sequence of floats to a 2‑D float32 tensor."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


# ----------------------------------------------------------------------
# Classical estimator core
# ----------------------------------------------------------------------
class FastHybridEstimator:
    """
    Evaluate either a Torch model or a Qiskit circuit for a list of
    parameter sets and observables.  Supports optional shot‑noise for
    classical models and shot sampling for quantum circuits.
    """

    def __init__(self, model: Union[nn.Module, QuantumCircuit]) -> None:
        self.model = model
        self._is_classical = isinstance(model, nn.Module)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def evaluate(
        self,
        observables: Iterable[Any],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float | complex]]:
        """
        Compute the expectation values for each observable and
        parameter set.

        Parameters
        ----------
        observables
            Functions (for Torch models) or qiskit.BaseOperator (for
            quantum circuits).
        parameter_sets
            Sequence of parameter tuples to bind to the model/circuit.
        shots
            Number of shots to use when sampling a quantum circuit
            or to add Gaussian noise to classical predictions.
        seed
            Random seed for reproducibility.
        """
        if self._is_classical:
            return self._evaluate_classical(observables, parameter_sets, shots, seed)
        else:
            return self._evaluate_quantum(observables, parameter_sets, shots, seed)

    # ------------------------------------------------------------------
    # Classical evaluation
    # ------------------------------------------------------------------
    def _evaluate_classical(
        self,
        observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]],
        parameter_sets: Sequence[Sequence[float]],
        shots: int | None,
        seed: int | None,
    ) -> List[List[float]]:
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = []

        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params)
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

        if shots is None:
            return results

        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in results:
            noisy_row = [
                float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row
            ]
            noisy.append(noisy_row)
        return noisy

    # ------------------------------------------------------------------
    # Quantum evaluation
    # ------------------------------------------------------------------
    def _evaluate_quantum(
        self,
        observables: Iterable[Operator],
        parameter_sets: Sequence[Sequence[float]],
        shots: int | None,
        seed: int | None,
    ) -> List[List[complex]]:
        results: List[List[complex]] = []

        backend = Aer.get_backend("statevector_simulator")
        for values in parameter_sets:
            bound = self.model.assign_parameters(
                dict(zip(self.model.parameters, values)), inplace=False
            )
            if shots is None:
                state = Statevector.from_instruction(bound)
                row = [state.expectation_value(obs) for obs in observables]
            else:
                # QASM simulation with sampling
                qasm_backend = Aer.get_backend("qasm_simulator")
                job = execute(
                    bound,
                    qasm_backend,
                    shots=shots,
                    seed_simulator=seed,
                )
                result = job.result()
                shots_counts = result.get_counts(bound)
                # Convert counts to probability distribution
                probs = {
                    tuple(int(bit) for bit in bitstring[::-1]): count / shots
                    for bitstring, count in shots_counts.items()
                }
                row = []
                for obs in observables:
                    exp = 0.0
                    for bitstring, prob in probs.items():
                        exp += prob * obs.data[np.array(bitstring).sum()]
                    row.append(exp)
            results.append(row)
        return results


# ----------------------------------------------------------------------
# Quanvolution (classical & quantum) helpers
# ----------------------------------------------------------------------
class QuanvolutionFilter(nn.Module):
    """
    Classical 2‑D convolution filter that emulates a 2×2 kernel.
    """
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv(x)
        return features.view(x.size(0), -1)


class QuanvolutionClassifier(nn.Module):
    """
    Hybrid classifier that stacks the QuanvolutionFilter with a linear head.
    """
    def __init__(self) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        self.linear = nn.Linear(4 * 14 * 14, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)
        logits = self.linear(features)
        return nn.functional.log_softmax(logits, dim=-1)


def build_quanvolution_classifier() -> nn.Module:
    """Convenience constructor for the classical quanvolution classifier."""
    return QuanvolutionClassifier()


# ----------------------------------------------------------------------
# Fraud‑detection (classical) helpers
# ----------------------------------------------------------------------
@dataclass
class FraudLayerParameters:
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]


def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))


def _layer_from_params(params: FraudLayerParameters, *, clip: bool) -> nn.Module:
    weight = torch.tensor(
        [
            [params.bs_theta, params.bs_phi],
            [params.squeeze_r[0], params.squeeze_r[1]],
        ],
        dtype=torch.float32,
    )
    bias = torch.tensor(params.phases, dtype=torch.float32)
    if clip:
        weight = weight.clamp(-5.0, 5.0)
        bias = bias.clamp(-5.0, 5.0)
    linear = nn.Linear(2, 2)
    with torch.no_grad():
        linear.weight.copy_(weight)
        linear.bias.copy_(bias)
    activation = nn.Tanh()
    scale = torch.tensor(params.displacement_r, dtype=torch.float32)
    shift = torch.tensor(params.displacement_phi, dtype=torch.float32)

    class Layer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = linear
            self.activation = activation
            self.register_buffer("scale", scale)
            self.register_buffer("shift", shift)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            outputs = self.activation(self.linear(inputs))
            outputs = outputs * self.scale + self.shift
            return outputs

    return Layer()


def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> nn.Sequential:
    """Create a sequential PyTorch model mirroring the layered structure."""
    modules = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)


# ----------------------------------------------------------------------
# Auto‑encoder helpers
# ----------------------------------------------------------------------
@dataclass
class AutoencoderConfig:
    input_dim: int
    latent_dim: int = 32
    hidden_dims: tuple[int, int] = (128, 64)
    dropout: float = 0.1


class AutoencoderNet(nn.Module):
    """A lightweight multilayer perceptron auto‑encoder."""
    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        encoder_layers = []
        in_dim = config.input_dim
        for hidden in config.hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, hidden))
            encoder_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                encoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        encoder_layers.append(nn.Linear(in_dim, config.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        in_dim = config.latent_dim
        for hidden in reversed(config.hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, hidden))
            decoder_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                decoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        decoder_layers.append(nn.Linear(in_dim, config.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.encoder(inputs)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(latents)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.decode(self.encode(inputs))


def Autoencoder(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: tuple[int, int] = (128, 64),
    dropout: float = 0.1,
) -> AutoencoderNet:
    """Factory that mirrors the quantum helper returning a configured network."""
    config = AutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
    )
    return AutoencoderNet(config)


def train_autoencoder(
    model: AutoencoderNet,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
) -> list[float]:
    """Simple reconstruction training loop returning the loss history."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = torch.utils.data.TensorDataset(_ensure_batch(data))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    history: list[float] = []

    for _ in range(epochs):
        epoch_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            reconstruction = model(batch)
            loss = loss_fn(reconstruction, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)
    return history


__all__ = [
    "FastHybridEstimator",
    "QuanvolutionFilter",
    "QuanvolutionClassifier",
    "build_quanvolution_classifier",
    "FraudLayerParameters",
    "build_fraud_detection_program",
    "AutoencoderConfig",
    "AutoencoderNet",
    "Autoencoder",
    "train_autoencoder",
]
