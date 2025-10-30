import numpy as np
import pennylane as qml
from dataclasses import dataclass
from typing import Sequence, List

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

class QCNNFraudHybrid:
    """
    Quantum hybrid model that merges a QCNN-inspired ansatz with a photonic fraud‑detection ansatz.
    The circuit accepts input data and a single parameter vector that contains both QCNN and photonic
    parameters.  It returns the expectation value of Pauli Z on wire 0.
    """

    def __init__(self, num_qubits: int = 8, num_photonic_layers: int = 3):
        self.num_qubits = num_qubits
        self.num_photonic_layers = num_photonic_layers
        self.device = qml.device("default.qubit", wires=num_qubits)
        self.qnode = qml.QNode(self._circuit, self.device)

    def _circuit(self, x: Sequence[float], params: Sequence[float]) -> float:
        # Split parameters
        qcnn_params = params[:42]
        photonic_params_flat = params[42:]
        # Feature map
        qml.AngleEmbedding(x, wires=range(self.num_qubits))
        # QCNN ansatz
        start = 0
        start = self._conv_layer(qcnn_params, start, pairs=[(0,1),(2,3),(4,5),(6,7)])
        start = self._pool_layer(qcnn_params, start, sources=[0,1,2,3], sinks=[4,5,6,7])
        start = self._conv_layer(qcnn_params, start, pairs=[(4,5),(6,7)])
        start = self._pool_layer(qcnn_params, start, sources=[0,1], sinks=[2,3])
        start = self._conv_layer(qcnn_params, start, pairs=[(6,7)])
        start = self._pool_layer(qcnn_params, start, sources=[0], sinks=[1])
        # Photonic layers on wires 6 and 7
        photonic_layers = self._parse_photonic_params(photonic_params_flat)
        for layer_params in photonic_layers:
            self._photonic_layer(layer_params, wires=[6,7])
        return qml.expval(qml.PauliZ(0))

    def _conv_circuit(self, params: Sequence[float], wire0: int, wire1: int) -> None:
        qml.RZ(-np.pi/2, wires=wire1)
        qml.CNOT(wires=[wire1, wire0])
        qml.RZ(params[0], wires=wire0)
        qml.RY(params[1], wires=wire1)
        qml.CNOT(wires=[wire0, wire1])
        qml.RY(params[2], wires=wire1)
        qml.CNOT(wires=[wire1, wire0])
        qml.RZ(np.pi/2, wires=wire0)

    def _pool_circuit(self, params: Sequence[float], wire0: int, wire1: int) -> None:
        qml.RZ(-np.pi/2, wires=wire1)
        qml.CNOT(wires=[wire1, wire0])
        qml.RZ(params[0], wires=wire0)
        qml.RY(params[1], wires=wire1)
        qml.CNOT(wires=[wire0, wire1])
        qml.RY(params[2], wires=wire1)

    def _conv_layer(
        self,
        params: Sequence[float],
        start: int,
        pairs: Sequence[tuple[int, int]],
    ) -> int:
        for wire0, wire1 in pairs:
            self._conv_circuit(params[start : start + 3], wire0, wire1)
            start += 3
        return start

    def _pool_layer(
        self,
        params: Sequence[float],
        start: int,
        sources: Sequence[int],
        sinks: Sequence[int],
    ) -> int:
        for src, snk in zip(sources, sinks):
            self._pool_circuit(params[start : start + 3], src, snk)
            start += 3
        return start

    def _photonic_layer(self, params: FraudLayerParameters, wires: Sequence[int]) -> None:
        qml.BSgate(params.bs_theta, params.bs_phi, wires=wires)
        for i, phase in enumerate(params.phases):
            qml.Rgate(phase, wires=wires[i])
        for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
            qml.Sgate(r, phi, wires=wires[i])
        qml.BSgate(params.bs_theta, params.bs_phi, wires=wires)
        for i, phase in enumerate(params.phases):
            qml.Rgate(phase, wires=wires[i])
        for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
            qml.Dgate(r, phi, wires=wires[i])
        for i, k in enumerate(params.kerr):
            qml.Kgate(k, wires=wires[i])

    def _parse_photonic_params(self, flat_params: Sequence[float]) -> List[FraudLayerParameters]:
        layers = []
        for i in range(0, len(flat_params), 14):
            bs_theta = flat_params[i]
            bs_phi = flat_params[i + 1]
            phases = (flat_params[i + 2], flat_params[i + 3])
            squeeze_r = (flat_params[i + 4], flat_params[i + 5])
            squeeze_phi = (flat_params[i + 6], flat_params[i + 7])
            displacement_r = (flat_params[i + 8], flat_params[i + 9])
            displacement_phi = (flat_params[i + 10], flat_params[i + 11])
            kerr = (flat_params[i + 12], flat_params[i + 13])
            layers.append(
                FraudLayerParameters(
                    bs_theta=bs_theta,
                    bs_phi=bs_phi,
                    phases=phases,
                    squeeze_r=squeeze_r,
                    squeeze_phi=squeeze_phi,
                    displacement_r=displacement_r,
                    displacement_phi=displacement_phi,
                    kerr=kerr,
                )
            )
        return layers

__all__ = ["FraudLayerParameters", "QCNNFraudHybrid"]
