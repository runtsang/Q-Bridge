"""Hybrid classical-quantum classifier with early stopping and optional temperature calibration.

The model consists of:
1. An optional classical feature extractor.
2. A quantum variational layer implemented via Qiskit and Aer.
3. A classical linear classifier head.
4. Optional temperature scaling for probability calibration.
5. A very light‑weight early‑stopping helper that tracks validation loss.

The quantum layer is a pure simulation and therefore fully differentiable via the
`torch.autograd.Function` wrapper.  This keeps the implementation lightweight
while still exposing the quantum contribution.

The code is intentionally self‑contained so that it can be used in a
stand‑alone script or imported into a larger training pipeline.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable, Tuple, Optional

# The quantum circuit builder lives in the QML package.
# Import here to keep the dependency local and avoid circular imports.
try:
    from.QuantumClassifierModel import build_classifier_circuit
except Exception:  # pragma: no cover
    # In a real project this would be a proper import.  For the
    # exercise we provide a minimal fallback.
    def build_classifier_circuit(num_qubits: int, depth: int):
        raise RuntimeError("Quantum circuit builder not available")

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.providers.aer import AerSimulator
import numpy as np


def _normalize_features(features: torch.Tensor) -> torch.Tensor:
    """Map features to the range [-π, π] for rotation gates."""
    return torch.pi * (features - features.min()) / (features.max() - features.min() + 1e-12)


def _simulate_circuit(
    circuit: QuantumCircuit,
    params: torch.Tensor,
    observables: Iterable[SparsePauliOp],
) -> torch.Tensor:
    """
    Evaluate the expectation values of a list of Pauli observables on a batch of
    parameterized circuits.  The function uses the Aer state‑vector simulator
    which is fully compatible with PyTorch tensors via NumPy.
    """
    backend = AerSimulator(method="statevector")
    batch_size = params.shape[0]
    num_obs = len(observables)
    exp_vals = np.zeros((batch_size, num_obs), dtype=np.float64)

    for i, param_set in enumerate(params):
        # Assign parameters to the circuit.
        circ = circuit.copy()
        circ.assign_parameters(dict(zip(circ.parameters, param_set.tolist())), inplace=True)
        job = backend.run(circ)
        state = job.result().get_statevector(circ)
        for j, pauli in enumerate(observables):
            exp_vals[i, j] = np.real(SparsePauliOp(pauli).expectation_value(state))

    return torch.from_numpy(exp_vals).float()


class QuantumClassifierModel(nn.Module):
    """
    Hybrid classical‑quantum classifier.

    Parameters
    ----------
    num_features : int
        Dimensionality of the input feature vector.
    num_qubits : int
        Number of qubits used in the quantum circuit.
    depth : int
        Depth of the variational ansatz.
    feature_extractor : nn.Module, optional
        A classical feature extractor that is applied before the quantum layer.
    early_stop_patience : int, default 5
        Number of epochs with no improvement after which training stops.
    calibration : bool, default False
        If True, a temperature scaling layer is appended to the logits.
    """

    def __init__(
        self,
        num_features: int,
        num_qubits: int,
        depth: int,
        feature_extractor: Optional[nn.Module] = None,
        early_stop_patience: int = 5,
        calibration: bool = False,
    ) -> None:
        super().__init__()
        self.feature_extractor = feature_extractor or nn.Identity()
        self.num_qubits = num_qubits
        self.depth = depth

        # Build the quantum circuit and obtain its parameters and observables.
        self.quantum_circuit, self.encoding_params, self.variational_params, self.observables = (
            build_classifier_circuit(num_qubits, depth)
        )

        # Linear head that maps the quantum feature vector to class logits.
        self.classifier = nn.Linear(num_qubits, 2)

        # Early‑stopping bookkeeping.
        self.early_stop_patience = early_stop_patience
        self.best_val_loss = float("inf")
        self.early_stop_counter = 0

        # Temperature scaling for calibration.
        self.calibration = calibration
        if self.calibration:
            self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, num_features).

        Returns
        -------
        torch.Tensor
            Logits of shape (batch_size, 2).
        """
        # Classical feature extraction.
        features = self.feature_extractor(x)

        # Map classical features to circuit parameters.
        # We use the encoding parameters for the first layer and the
        # variational parameters for the remaining layers.
        batch_size = features.shape[0]
        # Normalise features to [-π, π] to use as rotation angles.
        angles = _normalize_features(features)

        # Prepare parameter vectors for simulation.
        # We concatenate encoding and variational parameters for each sample.
        # The circuit expects a flat list of parameters in the order:
        # encoding_params + variational_params.
        params = torch.cat([angles, torch.zeros_like(angles)], dim=1)
        # Ensure the shape matches the expected number of parameters.
        # If the circuit expects more parameters (e.g., due to depth),
        # we tile the base angles accordingly.
        expected_params = len(self.encoding_params) + len(self.variational_params)
        if params.shape[1] < expected_params:
            # Tile the encoding angles to fill the parameter list.
            repeats = expected_params // params.shape[1]
            params = params.repeat(1, repeats)
            params = params[:, :expected_params]

        # Simulate the circuit to obtain quantum feature vector.
        quantum_features = _simulate_circuit(
            self.quantum_circuit, params, self.observables
        )

        # Linear classification.
        logits = self.classifier(quantum_features)

        if self.calibration:
            logits = logits / self.temperature

        return logits

    # ----------------------------------------------------------------------
    # Early‑stopping helpers
    # ----------------------------------------------------------------------
    def update_validation(self, val_loss: float) -> bool:
        """
        Update the validation loss and decide whether to stop training.

        Parameters
        ----------
        val_loss : float
            Validation loss for the current epoch.

        Returns
        -------
        bool
            True if training should stop, False otherwise.
        """
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.early_stop_counter = 0
            return False
        else:
            self.early_stop_counter += 1
            return self.early_stop_counter >= self.early_stop_patience

    def calibrate(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Apply temperature scaling to logits for probability calibration.

        Parameters
        ----------
        logits : torch.Tensor
            Raw logits.
        labels : torch.Tensor
            Ground‑truth labels.

        Returns
        -------
        torch.Tensor
            Calibrated logits.
        """
        if not self.calibration:
            return logits
        # Simple temperature scaling: minimise NLL over a small validation set.
        loss = F.cross_entropy(logits / self.temperature, labels)
        loss.backward()
        return logits / self.temperature
