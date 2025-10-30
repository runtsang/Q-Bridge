import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf
import numpy as np
import qiskit
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN as EstimatorQNNClass
from qiskit.primitives import StatevectorEstimator as Estimator

class QuantumNATModel(tq.QuantumModule):
    """
    Quantum hybrid model that mirrors the classical QuantumNATModel.
    It uses a torchquantum encoder, a variational layer, a quantum
    self‑attention circuit (implemented with Qiskit) and a measurement head.
    """
    def __init__(self, num_classes: int = 4, embed_dim: int = 4):
        super().__init__()
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.q_layer = self.QLayer()
        self.attn_module = self._build_quantum_attention()
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(self.n_wires, num_classes)
        self.norm = nn.BatchNorm1d(num_classes)

    class QLayer(tq.QuantumModule):
        def __init__(self):
            super().__init__()
            self.n_wires = 4
            self.random_layer = tq.RandomLayer(n_ops=50, wires=list(range(self.n_wires)))
            self.rx0 = tq.RX(has_params=True, trainable=True)
            self.ry0 = tq.RY(has_params=True, trainable=True)
            self.rz0 = tq.RZ(has_params=True, trainable=True)
            self.crx0 = tq.CRX(has_params=True, trainable=True)

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice):
            self.random_layer(qdev)
            self.rx0(qdev, wires=0)
            self.ry0(qdev, wires=1)
            self.rz0(qdev, wires=3)
            self.crx0(qdev, wires=[0, 2])
            tqf.hadamard(qdev, wires=3, static=self.static_mode, parent_graph=self.graph)
            tqf.sx(qdev, wires=2, static=self.static_mode, parent_graph=self.graph)
            tqf.cnot(qdev, wires=[3, 0], static=self.static_mode, parent_graph=self.graph)

    def _build_quantum_attention(self):
        class QuantumSelfAttention:
            def __init__(self, n_qubits: int):
                self.n_qubits = n_qubits
                self.backend = Aer.get_backend("qasm_simulator")

            def run(self, rotation_params: np.ndarray,
                    entangle_params: np.ndarray,
                    shots: int = 1024) -> dict:
                circuit = QuantumCircuit(self.n_qubits, self.n_qubits)
                for i in range(self.n_qubits):
                    circuit.rx(rotation_params[3 * i], i)
                    circuit.ry(rotation_params[3 * i + 1], i)
                    circuit.rz(rotation_params[3 * i + 2], i)
                for i in range(self.n_qubits - 1):
                    circuit.crx(entangle_params[i], i, i + 1)
                circuit.measure(range(self.n_qubits), range(self.n_qubits))
                job = execute(circuit, self.backend, shots=shots)
                result = job.result()
                return result.get_counts(circuit)
        return QuantumSelfAttention(self.n_wires)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True)
        pooled = torch.nn.functional.avg_pool2d(x, 6).view(bsz, 16)
        self.encoder(qdev, pooled)
        self.q_layer(qdev)
        features = self.measure(qdev)
        # Run the quantum self‑attention circuit with random parameters
        rot_params = np.random.randn(12)
        ent_params = np.random.randn(3)
        attn_counts = self.attn_module.run(rot_params, ent_params)
        # Convert counts dictionary to a dense tensor (placeholder logic)
        attn_tensor = torch.tensor([0.0] * self.n_wires, device=x.device, dtype=torch.float32)
        out = self.head(features + attn_tensor)
        return self.norm(out)

# ---------------------------------------------------------------------------

def generate_superposition_data(num_wires: int, samples: int):
    """
    Quantum superposition state generator used in the regression example.
    """
    omega_0 = np.zeros(2 ** num_wires, dtype=complex)
    omega_0[0] = 1.0
    omega_1 = np.zeros(2 ** num_wires, dtype=complex)
    omega_1[-1] = 1.0
    thetas = 2 * np.pi * np.random.rand(samples)
    phis = 2 * np.pi * np.random.rand(samples)
    states = np.zeros((samples, 2 ** num_wires), dtype=complex)
    for i in range(samples):
        states[i] = np.cos(thetas[i]) * omega_0 + np.exp(1j * phis[i]) * np.sin(thetas[i]) * omega_1
    labels = np.sin(2 * thetas) * np.cos(phis)
    return states, labels

class RegressionDataset(torch.utils.data.Dataset):
    """
    Dataset wrapper that returns quantum state tensors and labels.
    """
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, idx: int):
        return {
            "states": torch.tensor(self.states[idx], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }

def EstimatorQNN():
    """
    Qiskit EstimatorQNN example wrapped in a convenient function.
    """
    params1 = [Parameter("input1"), Parameter("weight1")]
    qc1 = QuantumCircuit(1)
    qc1.h(0)
    qc1.ry(params1[0], 0)
    qc1.rx(params1[1], 0)

    observable1 = SparsePauliOp.from_list([("Y" * qc1.num_qubits, 1)])

    estimator = Estimator()
    estimator_qnn = EstimatorQNNClass(
        circuit=qc1,
        observables=observable1,
        input_params=[params1[0]],
        weight_params=[params1[1]],
        estimator=estimator,
    )
    return estimator_qnn
