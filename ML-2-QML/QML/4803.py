import pennylane as qml
import numpy as np
from dataclasses import dataclass
from typing import Iterable, Tuple

@dataclass
class PhotonicLayerParams:
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

def _photonic_circuit(params: PhotonicLayerParams, clip: bool = False):
    dev = qml.device('default.qubit', wires=2)
    @qml.qnode(dev)
    def circuit(inputs: np.ndarray):
        qml.RX(inputs[0], wires=0)
        qml.RX(inputs[1], wires=1)
        qml.CSWAP(0, 1, 0)
        qml.PhaseShift(params.phases[0], wires=0)
        qml.PhaseShift(params.phases[1], wires=1)
        qml.RX(_clip(params.squeeze_r[0], 5), wires=0)
        qml.RX(_clip(params.squeeze_r[1], 5), wires=1)
        qml.RY(_clip(params.displacement_r[0], 5), wires=0)
        qml.RY(_clip(params.displacement_r[1], 5), wires=1)
        qml.CPhase(_clip(params.kerr[0], 1), wires=[0,1])
        return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))
    return circuit

def build_fraud_detection_program(input_params: PhotonicLayerParams,
                                  layers: Iterable[PhotonicLayerParams]):
    dev = qml.device('default.qubit', wires=2)
    @qml.qnode(dev)
    def circuit(inputs: np.ndarray):
        out0, out1 = _photonic_circuit(input_params, clip=False)(inputs)
        for params in layers:
            out0, out1 = _photonic_circuit(params, clip=True)([out0, out1])
        return (out0 + out1) / 2
    return circuit

class ConvFilterQuantum:
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0, shots=1024):
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.shots = shots
        self.wires = list(range(kernel_size ** 2))
        self.dev = qml.device('default.qubit', wires=self.wires)

    @qml.qnode
    def _circuit(self, data: np.ndarray):
        for i, val in enumerate(data):
            angle = np.pi if val > self.threshold else 0.0
            qml.RX(angle, wires=i)
        for i in range(0, len(self.wires)-1, 2):
            qml.CNOT(self.wires[i], self.wires[i+1])
        probs = [qml.probs(w) for w in self.wires]
        return sum([p[1] for p in probs]) / len(self.wires)

    def run(self, data: np.ndarray) -> float:
        flat = data.reshape(-1)
        return self._circuit(flat)

class QLSTMCellQuantum:
    def __init__(self, input_dim: int, hidden_dim: int, n_wires: int, shots=1024):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_wires = n_wires
        self.shots = shots
        self.dev = qml.device('default.qubit', wires=n_wires)
        self.wires = list(range(n_wires))

    def _gate(self, params: np.ndarray):
        @qml.qnode(self.dev)
        def circuit(x):
            for i, val in enumerate(x):
                qml.RX(val, wires=i)
            for i in range(0, self.n_wires-1, 2):
                qml.CNOT(self.wires[i], self.wires[i+1])
            return qml.expval(qml.PauliZ(0))
        return circuit(params)

    def forward(self, x: np.ndarray, hx: np.ndarray, cx: np.ndarray):
        f_raw = np.dot(np.concatenate([x, hx]), np.random.randn(x.size + hx.size, self.n_wires))
        i_raw = np.dot(np.concatenate([x, hx]), np.random.randn(x.size + hx.size, self.n_wires))
        g_raw = np.dot(np.concatenate([x, hx]), np.random.randn(x.size + hx.size, self.n_wires))
        o_raw = np.dot(np.concatenate([x, hx]), np.random.randn(x.size + hx.size, self.n_wires))

        f = self._gate(f_raw)
        i = self._gate(i_raw)
        g = self._gate(g_raw)
        o = self._gate(o_raw)

        cx = f * cx + i * g
        hx = o * np.tanh(cx)
        return hx, cx

class FraudDetectionHybrid:
    def __init__(self,
                 photonic_params: Iterable[PhotonicLayerParams],
                 conv_kernel: int = 2,
                 n_qubits: int = 4,
                 hidden_dim: int = 16):
        self.photonic_circuit = build_fraud_detection_program(photonic_params[0], photonic_params[1:])
        self.conv_filter = ConvFilterQuantum(kernel_size=conv_kernel)
        self.lstm = QLSTMCellQuantum(input_dim=1, hidden_dim=hidden_dim, n_wires=n_qubits)

    def run(self, x: np.ndarray) -> np.ndarray:
        y = self.photonic_circuit(x)
        y_img = np.full((self.conv_filter.kernel_size, self.conv_filter.kernel_size), y)
        conv_out = self.conv_filter.run(y_img)
        hx = np.zeros(self.lstm.hidden_dim)
        cx = np.zeros(self.lstm.hidden_dim)
        hx, cx = self.lstm.forward(np.array([conv_out]), hx, cx)
        return hx
