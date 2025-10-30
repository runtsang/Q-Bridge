import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchquantum as tq
import torchquantum.functional as tqf
import qiskit
from qiskit.circuit import ParameterVector
from qiskit_machine_learning.neural_networks import EstimatorQNN, SamplerQNN
from qiskit.primitives import StatevectorEstimator, StatevectorSampler
from qiskit.quantum_info import SparsePauliOp

# ------------------------------------------------------------------
# Quantum LSTM – gates implemented as small quantum circuits
# ------------------------------------------------------------------
class QuantumQLSTM(nn.Module):
    class QGate(tq.QuantumModule):
        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            # Simple RX encoder that maps classical inputs to qubit rotations
            self.encoder = tq.GeneralEncoder([
                {"input_idx": [0], "func": "rx", "wires": [0]},
                {"input_idx": [1], "func": "rx", "wires": [1]},
                {"input_idx": [2], "func": "rx", "wires": [2]},
                {"input_idx": [3], "func": "rx", "wires": [3]},
            ])
            self.params = nn.ModuleList(
                [tq.RX(has_params=True, trainable=True) for _ in range(n_wires)]
            )
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=x.shape[0], device=x.device)
            self.encoder(qdev, x)
            for wire, gate in enumerate(self.params):
                gate(qdev, wires=wire)
            # Entangle all qubits in a ring
            for wire in range(self.n_wires - 1):
                tqf.cnot(qdev, wires=[wire, wire + 1])
            tqf.cnot(qdev, wires=[self.n_wires - 1, 0])
            return self.measure(qdev)

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        self.forget = self.QGate(n_qubits)
        self.input = self.QGate(n_qubits)
        self.update = self.QGate(n_qubits)
        self.output = self.QGate(n_qubits)

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
        outputs = torch.cat(outputs, dim=0)
        return outputs, (hx, cx)

    def _init_states(self, inputs, states):
        if states is not None:
            return states
        batch = inputs.size(1)
        device = inputs.device
        return torch.zeros(batch, self.hidden_dim, device=device), torch.zeros(batch, self.hidden_dim, device=device)

# ------------------------------------------------------------------
# Quantum QCNN – ansatz from the reference implementation
# ------------------------------------------------------------------
class QuantumQCNN(nn.Module):
    """
    A quantum convolutional neural network that implements the
    multi‑layer ansatz described in the reference.  It returns
    a single expectation value per input sample.
    """
    def __init__(self, num_qubits: int = 8):
        super().__init__()
        self.num_qubits = num_qubits
        self.estimator = StatevectorEstimator()
        self.circuit = self._build_ansatz()
        obs = SparsePauliOp.from_list([("Z" + "I" * (num_qubits - 1), 1)])
        self.qnn = EstimatorQNN(
            circuit=self.circuit.decompose(),
            observables=obs,
            input_params=self._feature_map().parameters,
            weight_params=self.circuit.parameters,
            estimator=self.estimator,
        )

    def _feature_map(self):
        return qiskit.circuit.library.ZFeatureMap(self.num_qubits)

    def _build_ansatz(self):
        qc = qiskit.QuantumCircuit(self.num_qubits)
        # Layer 1
        qc.compose(self._conv_layer(self.num_qubits, "c1"), range(self.num_qubits), inplace=True)
        # Pool 1
        qc.compose(self._pool_layer([0,1,2,3], [4,5,6,7], "p1"), range(self.num_qubits), inplace=True)
        # Layer 2
        qc.compose(self._conv_layer(4, "c2"), range(4, 8), inplace=True)
        # Pool 2
        qc.compose(self._pool_layer([0,1], [2,3], "p2"), range(4, 8), inplace=True)
        # Layer 3
        qc.compose(self._conv_layer(2, "c3"), range(6, 8), inplace=True)
        # Pool 3
        qc.compose(self._pool_layer([0], [1], "p3"), range(6, 8), inplace=True)
        return qc

    def _conv_layer(self, n, prefix):
        qc = qiskit.QuantumCircuit(n, name="conv")
        params = ParameterVector(prefix, length=n*3)
        for i in range(0, n, 2):
            sub = self._conv_circuit(params[i*3:(i+2)*3])
            qc.append(sub, [i, i+1])
        return qc

    def _conv_circuit(self, params):
        qc = qiskit.QuantumCircuit(2)
        qc.rz(-np.pi/2, 1)
        qc.cx(1,0)
        qc.rz(params[0],0)
        qc.ry(params[1],1)
        qc.cx(0,1)
        qc.ry(params[2],1)
        qc.cx(1,0)
        qc.rz(np.pi/2,0)
        return qc

    def _pool_layer(self, sources, sinks, prefix):
        qc = qiskit.QuantumCircuit(len(sources)+len(sinks))
        params = ParameterVector(prefix, length=len(sources)*3)
        for src, snk, p in zip(sources, sinks, params):
            sub = self._pool_circuit(p)
            qc.append(sub, [src, snk])
        return qc

    def _pool_circuit(self, p):
        qc = qiskit.QuantumCircuit(2)
        qc.rz(-np.pi/2,1)
        qc.cx(1,0)
        qc.rz(p[0],0)
        qc.ry(p[1],1)
        qc.cx(0,1)
        qc.ry(p[2],1)
        return qc

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.qnn(inputs)

# ------------------------------------------------------------------
# Quantum Fully‑Connected Layer – parameterised circuit
# ------------------------------------------------------------------
class QuantumFCL(nn.Module):
    """
    A single‑qubit circuit that measures the expectation of Z.
    The circuit is parameterised by a single angle theta.
    """
    def __init__(self, n_qubits: int = 1):
        super().__init__()
        self.n_qubits = n_qubits
        self.backend = qiskit.Aer.get_backend("qasm_simulator")
        self.shots = 100
        self.circuit = qiskit.QuantumCircuit(n_qubits)
        self.theta = qiskit.circuit.Parameter("theta")
        self.circuit.h(range(n_qubits))
        self.circuit.barrier()
        self.circuit.ry(self.theta, range(n_qubits))
        self.circuit.measure_all()

    def run(self, thetas: np.ndarray) -> np.ndarray:
        job = qiskit.execute(
            self.circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=[{self.theta: theta} for theta in thetas],
        )
        result = job.result().get_counts(self.circuit)
        counts = np.array(list(result.values()))
        states = np.array(list(result.keys())).astype(float)
        probs = counts / self.shots
        expectation = np.sum(states * probs)
        return np.array([expectation])

# ------------------------------------------------------------------
# Quantum Sampler Network – uses qiskit SamplerQNN
# ------------------------------------------------------------------
class QuantumSamplerQNN(nn.Module):
    """
    Quantum sampler that implements a 2‑qubit circuit with
    trainable rotation angles and a classical post‑processing
    layer that outputs a 2‑class probability distribution.
    """
    def __init__(self):
        super().__init__()
        inputs = ParameterVector("input", 2)
        weights = ParameterVector("weight", 4)
        qc = qiskit.QuantumCircuit(2)
        qc.ry(inputs[0], 0)
        qc.ry(inputs[1], 1)
        qc.cx(0,1)
        qc.ry(weights[0], 0)
        qc.ry(weights[1], 1)
        qc.cx(0,1)
        qc.ry(weights[2], 0)
        qc.ry(weights[3], 1)
        sampler = StatevectorSampler()
        self.sampler_qnn = SamplerQNN(
            circuit=qc,
            input_params=inputs,
            weight_params=weights,
            sampler=sampler,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.sampler_qnn(inputs)

# ------------------------------------------------------------------
# Hybrid quantum‑classical tagger – unified interface
# ------------------------------------------------------------------
class QLSTMEnhancedQML(nn.Module):
    """
    Quantum‑enhanced LSTM tagger that mirrors the classical
    :class:`QLSTMEnhanced` but replaces each sub‑module with its
    quantum counterpart.  The model can be instantiated once
    and used in the same way as the classical version.
    """
    def __init__(self,
                 embedding_dim: int,
                 hidden_dim: int,
                 vocab_size: int,
                 tagset_size: int,
                 n_qubits: int = 4,
                 use_qcln: bool = True,
                 use_qfcl: bool = True,
                 use_qsampler: bool = True):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = QuantumQLSTM(embedding_dim, hidden_dim, n_qubits=n_qubits)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.cnn = QuantumQCNN(num_qubits=n_qubits)
        self.fcl = QuantumFCL(n_qubits) if use_qfcl else None
        self.sampler = QuantumSamplerQNN() if use_qsampler else None

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        cnn_out = self.cnn(lstm_out.view(len(sentence), -1))
        logits = self.hidden2tag(cnn_out)
        if self.fcl is not None:
            logits = torch.tensor(self.fcl.run(logits.detach().cpu().numpy()))
        if self.sampler is not None:
            logits = self.sampler(logits)
        return F.log_softmax(logits, dim=1)

__all__ = ["QLSTMEnhancedQML", "QuantumQLSTM", "QuantumQCNN", "QuantumFCL", "QuantumSamplerQNN"]
