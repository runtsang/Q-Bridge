import pennylane as qml
import pennylane.numpy as np
from dataclasses import dataclass
from typing import Iterable, Tuple, List

# --------------------------------------------------------------------------- #
# 1. Parameter‑shaped photonic circuit for fraud detection
# --------------------------------------------------------------------------- #
@dataclass
class FraudParams:
    bs_theta: float
    bs_phi: float
    phases: Tuple[float, float]
    squeeze_r: Tuple[float, float]
    squeeze_phi: Tuple[float, float]
    displacement_r: Tuple[float, float]
    displacement_phi: Tuple[float, float]
    kerr: Tuple[float, float]

# --------------------------------------------------------------------------- #
# 2. Variational quantum circuit that emulates the photonic layer
# --------------------------------------------------------------------------- #
class FraudQuantumCircuit:
    """
    A quantum circuit that maps the photonic parameters onto a
    two‑qubit variational ansatz.  The circuit is designed to be
    differentiable and can be used as a submodule in hybrid models.
    """
    def __init__(self, n_wires: int = 2):
        self.dev = qml.device("default.qubit", wires=n_wires)

    def circuit(self, params: FraudParams, x: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        params : FraudParams
            Photonic‑style parameters.
        x : array_like
            2‑dimensional input vector encoded into qubit amplitudes.
        Returns
        -------
        expectation : array
            Expectation values of PauliZ on both qubits.
        """
        @qml.qnode(self.dev, interface="autograd")
        def circuit(x):
            # Encode classical input as rotation angles
            qml.RX(x[0], wires=0)
            qml.RY(x[1], wires=1)

            # Beam splitter‑like entanglement
            qml.CZ(wires=[0, 1])

            # Parameterized rotations mimicking squeezed‑displacement
            qml.RX(params.bs_theta, wires=0)
            qml.RY(params.bs_phi, wires=1)
            qml.RZ(params.phases[0], wires=0)
            qml.RZ(params.phases[1], wires=1)

            # Squeezing‑style gates via parameterised rotations
            qml.RX(params.squeeze_r[0], wires=0)
            qml.RY(params.squeeze_r[1], wires=1)
            qml.RZ(params.squeeze_phi[0], wires=0)
            qml.RZ(params.squeeze_phi[1], wires=1)

            # Displacement‑style operations
            qml.RX(params.displacement_r[0], wires=0)
            qml.RY(params.displacement_r[1], wires=1)
            qml.RZ(params.displacement_phi[0], wires=0)
            qml.RZ(params.displacement_phi[1], wires=1)

            # Kerr‑like nonlinearity via ZZ interaction
            qml.CZ(wires=[0, 1])

            return [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))]

        return circuit(x)

    def __call__(self, params: FraudParams, x: np.ndarray) -> np.ndarray:
        return self.circuit(params, x)

# --------------------------------------------------------------------------- #
# 3. Hybrid quantum‑classical wrapper
# --------------------------------------------------------------------------- #
class HybridFraudDetector:
    """
    A hybrid model that combines the classical transformer backbone
    (from the ML module) with the quantum fraud circuit.
    """
    def __init__(self, params: Iterable[FraudParams], n_wires: int = 2):
        self.quantum_circuit = FraudQuantumCircuit(n_wires=n_wires)
        self.params = list(params)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        x : (batch, seq_len, 2)
        Returns a fraud score per example.
        """
        scores = []
        for sample in x:
            # Flatten sample to (seq_len, 2) and sum quantum outputs
            sample_scores = sum(self.quantum_circuit(self.params[0], inp) for inp in sample)
            scores.append(sample_scores.sum())
        return np.array(scores)

__all__ = [
    "FraudParams",
    "FraudQuantumCircuit",
    "HybridFraudDetector",
]
