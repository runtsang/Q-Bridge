import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute
from qiskit.circuit.random import random_circuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.circuit.library import ZFeatureMap
import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf

class QuantumConvFilter:
    """Quantum convolution filter (quanvolution) based on Qiskit."""
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0, backend=None, shots: int = 100):
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.n_qubits = kernel_size ** 2
        self.backend = backend or qiskit.Aer.get_backend("qasm_simulator")
        self.shots = shots
        self._circuit = QuantumCircuit(self.n_qubits)
        self.theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            self._circuit.rx(self.theta[i], i)
        self._circuit.barrier()
        self._circuit += random_circuit(self.n_qubits, 2)
        self._circuit.measure_all()

    def run(self, data: np.ndarray) -> float:
        # data shape: (kernel_size, kernel_size)
        data = data.flatten()
        param_binds = []
        for val in data:
            bind = {theta: np.pi if val > self.threshold else 0 for theta in self.theta}
            param_binds.append(bind)
        job = execute(self._circuit, self.backend, shots=self.shots, parameter_binds=param_binds)
        result = job.result()
        counts = result.get_counts(self._circuit)
        counts_sum = 0
        for key, val in counts.items():
            ones = sum(int(bit) for bit in key)
            counts_sum += ones * val
        return counts_sum / (self.shots * self.n_qubits)

class QuantumSelfAttention:
    """Quantum self‑attention block implemented with Qiskit."""
    def __init__(self, n_qubits: int = 4, backend=None):
        self.n_qubits = n_qubits
        self.backend = backend or qiskit.Aer.get_backend("qasm_simulator")
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")

    def _build_circuit(self, rotation_params: np.ndarray, entangle_params: np.ndarray) -> QuantumCircuit:
        circuit = QuantumCircuit(self.qr, self.cr)
        for i in range(self.n_qubits):
            circuit.rx(rotation_params[3 * i], i)
            circuit.ry(rotation_params[3 * i + 1], i)
            circuit.rz(rotation_params[3 * i + 2], i)
        for i in range(self.n_qubits - 1):
            circuit.crx(entangle_params[i], i, i + 1)
        circuit.measure(self.qr, self.cr)
        return circuit

    def run(self, rotation_params: np.ndarray, entangle_params: np.ndarray, shots: int = 1024) -> np.ndarray:
        circuit = self._build_circuit(rotation_params, entangle_params)
        job = execute(circuit, self.backend, shots=shots)
        result = job.result()
        counts = result.get_counts(circuit)
        # Convert counts to probability distribution over bitstrings
        probs = np.array([counts.get(format(i, f'0{self.n_qubits}b'), 0) / shots for i in range(2**self.n_qubits)])
        return probs

class QuantumQCNN:
    """Quantum convolutional neural network using Qiskit Machine Learning."""
    def __init__(self, n_qubits: int = 8):
        self.n_qubits = n_qubits
        self.estimator = Estimator()
        self._build_ansatz()

    def _build_ansatz(self):
        # Build convolution and pooling layers as in the reference
        def conv_circuit(params):
            qc = QuantumCircuit(2)
            qc.rz(-np.pi/2, 1)
            qc.cx(1, 0)
            qc.rz(params[0], 0)
            qc.ry(params[1], 1)
            qc.cx(0, 1)
            qc.ry(params[2], 1)
            qc.cx(1, 0)
            qc.rz(np.pi/2, 0)
            return qc

        def conv_layer(num_qubits, param_prefix):
            qc = QuantumCircuit(num_qubits)
            params = ParameterVector(param_prefix, length=num_qubits * 3)
            for i in range(0, num_qubits, 2):
                qc.append(conv_circuit(params[i:i+3]), [i, i+1])
            return qc

        def pool_circuit(params):
            qc = QuantumCircuit(2)
            qc.rz(-np.pi/2, 1)
            qc.cx(1, 0)
            qc.rz(params[0], 0)
            qc.ry(params[1], 1)
            qc.cx(0, 1)
            qc.ry(params[2], 1)
            return qc

        def pool_layer(sources, sinks, param_prefix):
            num_qubits = len(sources) + len(sinks)
            qc = QuantumCircuit(num_qubits)
            params = ParameterVector(param_prefix, length=num_qubits//2 * 3)
            for src, sink in zip(sources, sinks):
                qc.append(pool_circuit(params[:3]), [src, sink])
                params = params[3:]
            return qc

        # Build full ansatz
        ansatz = QuantumCircuit(self.n_qubits)
        # Layer 1
        ansatz.append(conv_layer(8, "c1"), range(8))
        # Layer 2
        ansatz.append(pool_layer([0,1,2,3], [4,5,6,7], "p1"), range(8))
        # Layer 3
        ansatz.append(conv_layer(4, "c2"), range(4,8))
        # Layer 4
        ansatz.append(pool_layer([0,1], [2,3], "p2"), range(4,8))
        # Layer 5
        ansatz.append(conv_layer(2, "c3"), range(6,8))
        # Layer 6
        ansatz.append(pool_layer([0], [1], "p3"), range(6,8))

        # Feature map
        feature_map = ZFeatureMap(self.n_qubits)
        self.circuit = QuantumCircuit(self.n_qubits)
        self.circuit.compose(feature_map, range(self.n_qubits), inplace=True)
        self.circuit.compose(ansatz, range(self.n_qubits), inplace=True)

        # Observable
        self.observable = SparsePauliOp.from_list([("Z" + "I" * (self.n_qubits - 1), 1)])

        # Estimator QNN
        self.qnn = EstimatorQNN(
            circuit=self.circuit.decompose(),
            observables=self.observable,
            input_params=feature_map.parameters,
            weight_params=ansatz.parameters,
            estimator=self.estimator
        )

    def run(self, data: np.ndarray) -> float:
        # data shape: (n_qubits,)
        if data.ndim == 2:
            data = data.reshape(-1)
        if len(data)!= self.n_qubits:
            raise ValueError(f"Input data must have {self.n_qubits} elements.")
        result = self.qnn.predict([data])
        return float(result[0])

class QuantumQLSTM(nn.Module):
    """Quantum LSTM cell using TorchQuantum."""
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int):
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
            self.params = nn.ModuleList([tq.RX(has_params=True, trainable=True) for _ in range(n_wires)])
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

    def __init__(self, input_dim: int, hidden_dim: int = None, n_qubits: int = 4):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = n_qubits
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

    def forward(self, inputs: torch.Tensor, states: tuple = None):
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
        return torch.cat(outputs, dim=0), (hx, cx)

    def _init_states(self, inputs: torch.Tensor, states: tuple = None):
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (torch.zeros(batch_size, self.hidden_dim, device=device),
                torch.zeros(batch_size, self.hidden_dim, device=device))

class HybridConvQLSTM:
    """Quantum‑classical hybrid pipeline that chains quanvolution, quantum self‑attention,
    quantum QCNN, and quantum LSTM."""
    def __init__(self,
                 kernel_size: int = 2,
                 threshold: float = 0.0,
                 embed_dim: int = 8,
                 hidden_dim: int = 32,
                 n_qubits: int = 4,
                 qc_n_qubits: int = 8):
        self.conv = QuantumConvFilter(kernel_size, threshold)
        self.attn = QuantumSelfAttention(n_qubits)
        self.qcnn = QuantumQCNN(qc_n_qubits)
        self.lstm = QuantumQLSTM(embed_dim, hidden_dim, n_qubits)

    def run(self, data: np.ndarray) -> torch.Tensor:
        """
        Parameters
        ----------
        data : np.ndarray
            Input data for the quantum convolution filter.
            For the self‑attention part we generate dummy parameters.
        Returns
        -------
        torch.Tensor
            Final hidden state from the quantum LSTM.
        """
        # Step 1: Quanvolution
        conv_out = self.conv.run(data)  # float

        # Step 2: Prepare dummy parameters for quantum self‑attention
        rotation_params = np.random.rand(self.attn.n_qubits * 3) * 2 * np.pi
        entangle_params  = np.random.rand(self.attn.n_qubits - 1) * 2 * np.pi

        # Step 3: Quantum self‑attention
        attn_probs = self.attn.run(rotation_params, entangle_params)  # (2^n,)
        # Collapse to a single feature vector (e.g., average probability)
        attn_feature = attn_probs.mean()

        # Step 4: Quantum QCNN
        qcnn_input = np.array([conv_out, attn_feature] + [0] * (self.qcnn.n_qubits - 2))
        qcnn_output = self.qcnn.run(qcnn_input)  # float

        # Step 5: Quantum LSTM
        lstm_input = torch.tensor([qcnn_output], dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1,1,1)
        lstm_out, _ = self.lstm(lstm_input)
        return lstm_out.squeeze()  # (hidden_dim,)

def Conv():
    """Factory returning the quantum hybrid module."""
    return HybridConvQLSTM()
