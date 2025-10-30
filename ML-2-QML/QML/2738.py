import pennylane as qml
import pennylane.numpy as np
import strawberryfields as sf
from strawberryfields.ops import BSgate, Rgate, Sgate, Dgate, Kgate
from dataclasses import dataclass
from typing import Tuple, Iterable

@dataclass
class FraudLayerParameters:
    bs_theta: float
    bs_phi: float
    phases: Tuple[float, float]
    squeeze_r: Tuple[float, float]
    squeeze_phi: Tuple[float, float]
    displacement_r: Tuple[float, float]
    displacement_phi: Tuple[float, float]
    kerr: Tuple[float, float]

def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))

def _apply_layer(modes, params: FraudLayerParameters, *, clip: bool) -> None:
    BSgate(params.bs_theta, params.bs_phi) | (modes[0], modes[1])
    for i, phase in enumerate(params.phases):
        Rgate(phase) | modes[i]
    for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
        Sgate(r if not clip else _clip(r, 5), phi) | modes[i]
    BSgate(params.bs_theta, params.bs_phi) | (modes[0], modes[1])
    for i, phase in enumerate(params.phases):
        Rgate(phase) | modes[i]
    for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
        Dgate(r if not clip else _clip(r, 5), phi) | modes[i]
    for i, k in enumerate(params.kerr):
        Kgate(k if not clip else _clip(k, 1)) | modes[i]

def build_fraud_detection_program(input_params: FraudLayerParameters,
                                  layers: Iterable[FraudLayerParameters]) -> sf.Program:
    program = sf.Program(2)
    with program.context as q:
        _apply_layer(q, input_params, clip=False)
        for layer in layers:
            _apply_layer(q, layer, clip=True)
    return program

class QuantumFraudFeatureExtractor:
    """Runs the photonic fraud‑detection circuit and returns photon‑number expectations."""
    def __init__(self,
                 input_params: FraudLayerParameters,
                 hidden_params: Iterable[FraudLayerParameters]) -> None:
        self.program = build_fraud_detection_program(input_params, hidden_params)

    def evaluate(self, shots: int = 1000) -> np.ndarray:
        eng = sf.Engine("default.qubit")
        result = eng.run(self.program, shots=shots)
        # Photon‑number expectation values for each mode
        exp = np.array([result.samples[:, i].mean() for i in range(2)])
        return exp

class QuantumFraudTransformer:
    """Variational transformer‑style classifier built with PennyLane."""
    def __init__(self, n_qubits: int = 2, n_layers: int = 2) -> None:
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dev = qml.device("default.qubit", wires=n_qubits)
        self.params = np.random.randn(n_layers, n_qubits)
        self.classifier = np.random.randn(n_qubits)

    def _ansatz(self, x: np.ndarray, params: np.ndarray):
        for i in range(self.n_layers):
            for j in range(self.n_qubits):
                qml.RX(x[j] + params[i, j], wires=j)
            for j in range(self.n_qubits - 1):
                qml.CNOT(wires=[j, j + 1])
            qml.CNOT(wires=[self.n_qubits - 1, 0])
        return [qml.expval(qml.PauliZ(j)) for j in range(self.n_qubits)]

    @qml.qnode
    def circuit(self, x: np.ndarray):
        return self._ansatz(x, self.params)

    def predict(self, x: np.ndarray) -> float:
        out = self.circuit(x)
        return np.tanh(np.dot(out, self.classifier))

class QuantumFraudTransformerHybrid:
    """Combines the photonic feature extractor with the variational transformer."""
    def __init__(self,
                 input_params: FraudLayerParameters,
                 hidden_params: Iterable[FraudLayerParameters],
                 transformer_cfg: dict) -> None:
        self.extractor = QuantumFraudFeatureExtractor(input_params, hidden_params)
        self.transformer = QuantumFraudTransformer(**transformer_cfg)

    def predict(self, shots: int = 1000) -> float:
        features = self.extractor.evaluate(shots=shots)
        return self.transformer.predict(features)
