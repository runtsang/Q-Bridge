"""Hybrid classical‑quantum auto‑encoder.

This module merges the classical MLP auto‑encoder from the original
seed with a lightweight quantum interface.  It also retains the
FastEstimator utilities from the second reference.

Typical usage::
    from Autoencoder__gen231 import AutoencoderHybrid
    model = AutoencoderHybrid(input_dim=784, use_quantum=True)
    loss_hist = train_autoencoder(model, data)
    probs = model.quantum_encode(data[:10])
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Iterable, List, Tuple, Callable, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# --------------------------------------------------------------------------- #
#   Classical Auto‑encoder – unchanged from seed                         #
# --------------------------------------------------------------------------- #
@dataclass
class AutoencoderConfig:
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
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
    hidden_dims: Tuple[int, int] = (128, 64),
    dropout: float = 0.1,
) -> AutoencoderNet:
    """Factory mirroring the original seed."""
    return AutoencoderNet(
        AutoencoderConfig(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )
    )

# --------------------------------------------------------------------------- #
#   Training helper – unchanged from the seed                         #
# --------------------------------------------------------------------------- #
def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    if tensor.dtype!= torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor

def train_autoencoder(
    model: nn.Module,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
) -> List[float]:
    """Simple reconstruction training loop returning the loss history."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = TensorDataset(_as_tensor(data))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    history: List[float] = []

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

# --------------------------------------------------------------------------- #
#   Quantum helper – optional, only imported when qiskit is available   #
# --------------------------------------------------------------------------- #
# The following imports are wrapped in a try/except so that the module can
# be used in environments without qiskit.  They are only required for the
# ``quantum_encode`` method.
try:
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
    from qiskit.circuit.library import RealAmplitudes
    from qiskit.primitives import StatevectorSampler as Sampler
    from qiskit.quantum_info import Statevector
except Exception:  # pragma: no cover
    QuantumCircuit = None  # type: ignore[assignment]
    RealAmplitudes = None
    Sampler = None
    Statevector = None

# --------------------------------------------------------------------------- #
#   Estimator utilities – copy of the original FastEstimator from ref 2   #
# --------------------------------------------------------------------------- #
ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]

def _ensure_batch(values: Iterable[float]) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor

class FastBaseEstimator:
    """Evaluate neural nets for batches of inputs and observables."""
    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Iterable[Iterable[float]],
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
        return results

class FastEstimator(FastBaseEstimator):
    """Adds optional Gaussian shot noise to the deterministic estimator."""
    def __init__(self, model: nn.Module, shots: int | None = None, seed: int | None = None):
        super().__init__(model)
        self.shots = shots
        self.seed = seed

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Iterable[Iterable[float]],
    ) -> List[List[float]]:
        raw = super().evaluate(observables, parameter_sets)
        if self.shots is None:
            return raw
        rng = np.random.default_rng(self.seed)
        noisy: List[List[float]] = []
        for row in raw:
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / self.shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy

# --------------------------------------------------------------------------- #
#   Hybrid class combining the above components                           #
# --------------------------------------------------------------------------- #
class AutoencoderHybrid(nn.Module):
    """
    A hybrid auto‑encoder that exposes both a classical MLP and an optional
    quantum auto‑encoder.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input data.
    latent_dim : int, default 32
        Size of the latent representation.
    hidden_dims : Tuple[int, int], default (128, 64)
        Sizes of the hidden layers in the MLP.
    dropout : float, default 0.1
        Dropout probability.
    use_quantum : bool, default False
        If True and qiskit is available, the model will expose a
        ``quantum_encode`` method that evaluates a quantum circuit.
    """

    def __init__(
        self,
        input_dim: int,
        *,
        latent_dim: int = 32,
        hidden_dims: Tuple[int, int] = (128, 64),
        dropout: float = 0.1,
        use_quantum: bool = False,
    ) -> None:
        super().__init__()
        self.classical = Autoencoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )
        self.use_quantum = use_quantum and QuantumCircuit is not None

    # ------------------------------------------------------------------ #
    #   Classical interface                                                #
    # ------------------------------------------------------------------ #
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Run the classical auto‑encoder."""
        return self.classical(inputs)

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        """Encode inputs to latent space."""
        return self.classical.encode(inputs)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latents back to input space."""
        return self.classical.decode(latents)

    # ------------------------------------------------------------------ #
    #   Quantum interface – available only when ``use_quantum=True``      #
    # ------------------------------------------------------------------ #
    def quantum_encode(self, inputs: torch.Tensor, *, shots: int | None = 1024) -> List[float]:
        """
        Encode a batch of inputs using the quantum auto‑encoder circuit.

        Parameters
        ----------
        inputs : torch.Tensor
            Batch of input vectors with shape ``(batch, input_dim)``.
        shots : int | None
            Number of shots for the state‑vector sampler.  ``None`` uses the exact
            state‑vector simulation.

        Returns
        -------
        List[float]
            List of measurement probabilities (or amplitudes) for each input.
        """
        if not self.use_quantum:
            raise RuntimeError("Quantum backend not available. Set `use_quantum=True` and install qiskit.")

        # Import quantum dependencies lazily
        from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
        from qiskit.circuit.library import RealAmplitudes
        from qiskit.primitives import StatevectorSampler as Sampler
        from qiskit.quantum_info import Statevector

        latent = self.classical.encode(inputs)
        latent_np = latent.detach().cpu().numpy()
        probs: List[float] = []
        for _ in latent_np:
            num_latent = latent.shape[1]
            num_trash = 1
            qr = QuantumRegister(num_latent + 2 * num_trash + 1, "q")
            cr = ClassicalRegister(1, "c")
            circuit = QuantumCircuit(qr, cr)
            circuit.compose(
                RealAmplitudes(num_latent + num_trash, reps=5),
                range(num_latent + num_trash),
                inplace=True,
            )
            circuit.barrier()
            aux = num_latent + 2 * num_trash
            circuit.h(aux)
            for i in range(num_trash):
                circuit.cswap(aux, num_latent + i, num_latent + num_trash + i)
            circuit.h(aux)
            circuit.measure(aux, cr[0])

            if shots is None:
                state = Statevector.from_instruction(circuit)
                probs.append(state.probabilities()[0].item())
            else:
                sampler = Sampler()
                result = sampler.run(circuit, shots=shots).result()
                probs.append(result.quasi_dists[0][0])
        return probs

    # ------------------------------------------------------------------ #
    #   Estimator utilities – compatible with the original FastEstimator    #
    # ------------------------------------------------------------------ #
    def estimator(self, use_shots: bool = False, shots: int = 1024) -> FastEstimator:
        """
        Return a :class:`FastEstimator` that operates on the classical part of the hybrid model.

        Parameters
        ----------
        use_shots : bool
            If ``True`` the estimator will add Gaussian noise to the
            deterministic outputs.
        shots : int
            Number of shots to use when adding noise.
        """
        if use_shots:
            return FastEstimator(self.classical, shots=shots)
        return FastBaseEstimator(self.classical)

__all__ = [
    "AutoencoderHybrid",
    "AutoencoderConfig",
    "AutoencoderNet",
    "Autoencoder",
    "train_autoencoder",
    "FastBaseEstimator",
    "FastEstimator",
]
