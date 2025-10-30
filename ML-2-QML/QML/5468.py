import torch
import torch.nn as nn
import torchquantum as tq
import strawberryfields as sf
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import ParameterVector
import numpy as np
import tq.functional as tqf
from typing import Tuple

# Photonic fraud detection parameters (renamed)
class PhotonicFraudParameters:
    def __init__(self, bs_theta, bs_phi, phases, squeeze_r, squeeze_phi, displacement_r, displacement_phi, kerr):
        self.bs_theta = bs_theta
        self.bs_phi = bs_phi
        self.phases = phases
        self.squeeze_r = squeeze_r
        self.squeeze_phi = squeeze_phi
        self.displacement_r = displacement_r
        self.displacement_phi = displacement_phi
        self.kerr = kerr

def _clip_val(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))

def _apply_photonic_layer(modes, params, *, clip: bool):
    sf.ops.BSgate(params.bs_theta, params.bs_phi) | (modes[0], modes[1])
    for i, phase in enumerate(params.phases):
        sf.ops.Rgate(phase) | modes[i]
    for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
        sf.ops.Sgate(r if not clip else _clip_val(r, 5), phi) | modes[i]
    sf.ops.BSgate(params.bs_theta, params.bs_phi) | (modes[0], modes[1])
    for i, phase in enumerate(params.phases):
        sf.ops.Rgate(phase) | modes[i]
    for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
        sf.ops.Dgate(r if not clip else _clip_val(r, 5), phi) | modes[i]
    for i, k in enumerate(params.kerr):
        sf.ops.Kgate(k if not clip else _clip_val(k, 1)) | modes[i]

def build_photonic_fraud_program(input_params, layers):
    program = sf.Program(2)
    with program.context as q:
        _apply_photonic_layer(q, input_params, clip=False)
        for layer in layers:
            _apply_photonic_layer(q, layer, clip=True)
    return program

# Quantum LSTM layer
class QLSTMQuantumLayer(tq.QuantumModule):
    """LSTM cell where gates are realised by small quantum circuits."""
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int) -> None:
            super().__init__()
            self.n_wires = n_wires
            self.encoder = tq.GeneralEncoder(
                [
                    {"input_idx": [0], "func": "rx", "wires": [0]},
                    {"input_idx": [1], "func": "rx", "wires": [1]},
                    {"input_idx": [2], "func": "rx", "wires": [2]},
                    {"input_idx": [3], "func": "rx", "wires": [3]},
                ]
            )
            self.params = nn.ModuleList(
                [tq.RX(has_params=True, trainable=True) for _ in range(n_wires)]
            )
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=x.shape[0], device=x.device)
            self.encoder(qdev, x)
            for wire, gate in enumerate(self.params):
                gate(qdev, wires=wire)
            for wire in range(self.n_wires):
                if wire == self.n_wires - 1:
                    tqf.cnot(qdev, wires=[wire, 0])
                else:
                    tqf.cnot(qdev, wires=[wire, wire + 1])
            return self.measure(qdev)

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        self.forget = self.QLayer(n_qubits)
        self.input = self.QLayer(n_qubits)
        self.update = self.QLayer(n_qubits)
        self.output = self.QLayer(n_qubits)

        self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)

    def forward(self, inputs: torch.Tensor, states: Tuple[torch.Tensor, torch.Tensor] | None = None):
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget(self.linear_forget(combined)))
            i = torch.sigmoid(self.input(self.linear_input(combined)))
            g = torch.tanh(self.update(self.linear_update(combined)))
            o = torch.sigmoid(self.output(self.linear_output(combined)))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        outputs = torch.cat(outputs, dim=0)
        return outputs, (hx, cx)

    def _init_states(
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (
            torch.zeros(batch_size, self.hidden_dim, device=device),
            torch.zeros(batch_size, self.hidden_dim, device=device),
        )

# QCNN quantum module
def conv_circuit(params):
    target = QuantumCircuit(2)
    target.rz(-np.pi / 2, 1)
    target.cx(1, 0)
    target.rz(params[0], 0)
    target.ry(params[1], 1)
    target.cx(0, 1)
    target.ry(params[2], 1)
    target.cx(1, 0)
    target.rz(np.pi / 2, 0)
    return target

def conv_layer(num_qubits, param_prefix):
    qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
    qubits = list(range(num_qubits))
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits * 3)
    for q1, q2 in zip(qubits[0::2], qubits[1::2]):
        qc = qc.compose(conv_circuit(params[param_index : (param_index + 3)]), [q1, q2])
        qc.barrier()
        param_index += 3
    for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
        qc = qc.compose(conv_circuit(params[param_index : (param_index + 3)]), [q1, q2])
        qc.barrier()
        param_index += 3

    qc_inst = qc.to_instruction()

    qc = QuantumCircuit(num_qubits)
    qc.append(qc_inst, qubits)
    return qc

def pool_circuit(params):
    target = QuantumCircuit(2)
    target.rz(-np.pi / 2, 1)
    target.cx(1, 0)
    target.rz(params[0], 0)
    target.ry(params[1], 1)
    target.cx(0, 1)
    target.ry(params[2], 1)
    return target

def pool_layer(sources, sinks, param_prefix):
    num_qubits = len(sources) + len(sinks)
    qc = QuantumCircuit(num_qubits, name="Pooling Layer")
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
    for source, sink in zip(sources, sinks):
        qc = qc.compose(pool_circuit(params[param_index : (param_index + 3)]), [source, sink])
        qc.barrier()
        param_index += 3

    qc_inst = qc.to_instruction()

    qc = QuantumCircuit(num_qubits)
    qc.append(qc_inst, range(num_qubits))
    return qc

class QCNNQuantum(tq.QuantumModule):
    """Quantum analogue of the classical QCNN model."""
    def __init__(self):
        super().__init__()
        self.n_qubits = 8
        self.circuit = conv_layer(8, "c1")
        self.backend = Aer.get_backend('statevector_simulator')
        self.q_device = tq.QuantumDevice(n_wires=self.n_qubits)

    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor) -> torch.Tensor:
        # Run convolutional layer
        job = execute(self.circuit, self.backend)
        result = job.result()
        statevector = result.get_statevector(self.circuit)
        z_exp = np.real(statevector[0]*np.conj(statevector[0]) - statevector[1]*np.conj(statevector[1]))
        return torch.tensor([z_exp], dtype=torch.float32)

# Quantum kernel
class QuantumKernel(tq.QuantumModule):
    """Quantum kernel evaluated via a fixed TorchQuantum ansatz."""
    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.ansatz = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        q_device.reset_states(x.shape[0])
        self.ansatz(q_device, x)
        self.ansatz(q_device, -y)
        return torch.abs(q_device.states.view(-1)[0])

# Hybrid fraud‑detection quantum model
class FraudDetectionHybrid(tq.QuantumModule):
    """
    Quantum‑centric fraud‑detection model that stitches together
    photonic circuits, a QCNN, a quantum LSTM and a quantum kernel.
    """
    def __init__(
        self,
        input_dim: int = 2,
        seq_dim: int = 1,
        hidden_dim: int = 32,
        n_qubits: int = 4,
        gamma: float = 1.0,
    ) -> None:
        super().__init__()
        # Photonic fraud circuit
        default_params = PhotonicFraudParameters(
            bs_theta=0.5,
            bs_phi=0.3,
            phases=(0.1, 0.2),
            squeeze_r=(0.2, 0.3),
            squeeze_phi=(0.4, 0.5),
            displacement_r=(0.6, 0.7),
            displacement_phi=(0.8, 0.9),
            kerr=(0.1, 0.2),
        )
        self.fraud_circuit = build_photonic_fraud_program(default_params, [])

        # QCNN quantum module
        self.cnn_quantum = QCNNQuantum()

        # Quantum LSTM
        self.lstm = QLSTMQuantumLayer(input_dim, hidden_dim, n_qubits)

        # Quantum kernel
        self.kernel = QuantumKernel()

        # Classical linear classifier
        self.classifier = nn.Linear(n_qubits + 3, 1)

    def forward(self, features: torch.Tensor, seq: torch.Tensor, feature_vec: torch.Tensor) -> torch.Tensor:
        # Photonic fraud measurement
        eng = sf.Engine('stabilizer')
        eng.run(self.fraud_circuit)
        photonic_val = eng.backend.get_expectation_value(sf.ops.Z(0) + sf.ops.Z(1))
        photonic_out = torch.tensor([photonic_val], dtype=torch.float32).repeat(features.size(0), 1)

        # QCNN quantum measurement
        qcnn_out = self.cnn_quantum(self.cnn_quantum.q_device, features)
        qcnn_out = qcnn_out.repeat(features.size(0), 1)

        # Quantum LSTM output
        seq_t = seq.transpose(0, 1)  # shape (seq_len, batch, seq_dim)
        lstm_out, _ = self.lstm(seq_t)
        lstm_last = lstm_out[-1]  # (batch, n_qubits)

        # Quantum kernel output
        kernel_out = self.kernel(self.kernel.q_device, feature_vec, feature_vec)
        kernel_out = kernel_out.repeat(features.size(0), 1)

        combined = torch.cat([photonic_out, qcnn_out, lstm_last, kernel_out], dim=1)
        logits = self.classifier(combined)
        return torch.sigmoid(logits)

__all__ = [
    "PhotonicFraudParameters",
    "build_photonic_fraud_program",
    "QLSTMQuantumLayer",
    "QCNNQuantum",
    "QuantumKernel",
    "FraudDetectionHybrid",
]
