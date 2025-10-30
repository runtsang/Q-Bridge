import pennylane as qml
import pennylane.numpy as np
from dataclasses import dataclass

@dataclass
class QuantumParams:
    bs_theta: float
    bs_phi: float
    ry0: float
    ry1: float
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    disp_r: tuple[float, float]
    disp_phi: tuple[float, float]
    kerr: tuple[float, float]

class HybridLayer:
    """Quantum photonic hybrid layer that mirrors the classical HybridLayer pipeline."""
    def __init__(self, n_modes: int = 2, cutoff_dim: int = 10, num_shots: int = 1024):
        self.n_modes = n_modes
        self.dev = qml.device("strawberryfields.fock", wires=n_modes, cutoff_dim=cutoff_dim)
        self.num_shots = num_shots
        self._circuit = self._build_circuit()

    def _build_circuit(self):
        @qml.qnode(self.dev, interface="autograd")
        def circuit(inputs: np.ndarray, params: QuantumParams):
            # Encode inputs as displacements on each mode
            for i in range(self.n_modes):
                qml.Displacement(inputs[i], 0.0, wires=i)

            # Quantum convolution filter: a beamsplitter between the two modes
            qml.BSgate(params.bs_theta, params.bs_phi, wires=[0, 1])

            # Sampler QNN: two variational rotations and a CNOT
            qml.RY(params.ry0, wires=0)
            qml.RY(params.ry1, wires=1)
            qml.CNOT(wires=[0, 1])

            # Photonic fraud‑style layer: squeezers, displacements, Kerr
            for i in range(self.n_modes):
                qml.Squeezing(params.squeeze_r[i], params.squeeze_phi[i], wires=i)
                qml.Displacement(params.disp_r[i], params.disp_phi[i], wires=i)
                qml.Kerr(params.kerr[i], wires=i)

            # Measurement: expectation of the Pauli‑Z observable on mode 0
            return qml.expval(qml.PauliZ(0))
        return circuit

    def run(self, inputs: np.ndarray, params: QuantumParams) -> np.ndarray:
        """Execute the quantum circuit and return the expectation value."""
        return self._circuit(inputs, params)

__all__ = ["HybridLayer"]
