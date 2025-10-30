import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN as QiskitEstimatorQNN
from qiskit.primitives import Estimator as QiskitEstimator

class FullyConnectedQuantumLayer:
    """Parameterised quantum circuit that outputs an expectation value."""
    def __init__(self, n_qubits: int = 1, shots: int = 1024):
        self.n_qubits = n_qubits
        self.shots = shots
        self.backend = Aer.get_backend("qasm_simulator")
        self.theta = Parameter("theta")
        self.circuit = QuantumCircuit(n_qubits)
        self.circuit.h(range(n_qubits))
        self.circuit.ry(self.theta, range(n_qubits))
        self.circuit.barrier()
        self.circuit.measure_all()

    def run(self, thetas):
        job = execute(
            self.circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=[{self.theta: theta} for theta in thetas],
        )
        result = job.result()
        counts = result.get_counts(self.circuit)
        expectation = 0.0
        total = 0
        for state, count in counts.items():
            val = int(state[::-1], 2)  # reverse bit order
            expectation += val * count
            total += count
        expectation /= total
        return np.array([expectation])

class QuantumEstimatorQNN:
    """Quantum EstimatorQNN that evaluates a simple circuit."""
    def __init__(self, n_qubits: int = 1, shots: int = 1024):
        self.params1 = [Parameter("input1"), Parameter("weight1")]
        self.qc1 = QuantumCircuit(n_qubits)
        self.qc1.h(0)
        self.qc1.ry(self.params1[0], 0)
        self.qc1.rx(self.params1[1], 0)
        self.qc1.measure_all()
        self.observable1 = SparsePauliOp.from_list([("Z" * n_qubits, 1)])
        self.estimator = QiskitEstimator()
        self.estimator_qnn = QiskitEstimatorQNN(
            circuit=self.qc1,
            observables=self.observable1,
            input_params=[self.params1[0]],
            weight_params=[self.params1[1]],
            estimator=self.estimator,
        )

    def evaluate(self, input_val: float, weight_val: float):
        result = self.estimator_qnn.evaluate(
            input_val=input_val,
            weight=weight_val,
        )
        return result[0]  # expectation value

class QuantumGateCircuit:
    """Simple parameterised circuit that acts as a quantum gate."""
    def __init__(self, n_qubits: int = 1, shots: int = 1024):
        self.n_qubits = n_qubits
        self.shots = shots
        self.backend = Aer.get_backend("qasm_simulator")
        self.theta = Parameter("theta")
        self.circuit = QuantumCircuit(n_qubits)
        self.circuit.h(range(n_qubits))
        self.circuit.rx(self.theta, range(n_qubits))
        self.circuit.barrier()
        self.circuit.measure_all()

    def run(self, theta: float):
        job = execute(
            self.circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=[{self.theta: theta}],
        )
        result = job.result()
        counts = result.get_counts(self.circuit)
        expectation = 0.0
        total = 0
        for state, count in counts.items():
            val = int(state[::-1], 2)
            expectation += val * count
            total += count
        expectation /= total
        return expectation

class QuantumLSTMCell:
    """LSTM cell where each gate is realised by a small quantum circuit."""
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int = 1):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        # Linear layers that map the concatenated input to gate parameters
        self.lin_forget = nn.Linear(input_dim + hidden_dim, 1)
        self.lin_input = nn.Linear(input_dim + hidden_dim, 1)
        self.lin_update = nn.Linear(input_dim + hidden_dim, 1)
        self.lin_output = nn.Linear(input_dim + hidden_dim, 1)

        # Quantum gate modules
        self.forget_gate = QuantumGateCircuit(n_qubits)
        self.input_gate = QuantumGateCircuit(n_qubits)
        self.update_gate = QuantumGateCircuit(n_qubits)
        self.output_gate = QuantumGateCircuit(n_qubits)

    def forward(self, x: torch.Tensor, hx: torch.Tensor, cx: torch.Tensor):
        combined = torch.cat([x, hx], dim=1)
        f_theta = self.lin_forget(combined).item()
        i_theta = self.lin_input(combined).item()
        g_theta = self.lin_update(combined).item()
        o_theta = self.lin_output(combined).item()

        f = self.forget_gate.run(f_theta)
        i = self.input_gate.run(i_theta)
        g = self.update_gate.run(g_theta)
        o = self.output_gate.run(o_theta)

        cx = f * cx + i * g
        hx = o * torch.tanh(cx)

        return hx, cx

class UnifiedQLSTM:
    """Hybrid classical‑quantum LSTM tagger that can switch between quantum and classical back‑ends."""
    def __init__(self, input_dim: int, hidden_dim: int, vocab_size: int, tagset_size: int,
                 n_qubits: int = 0, use_quantum: bool = False):
        self.use_quantum = use_quantum
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, input_dim)
        if use_quantum and n_qubits > 0:
            self.lstm_cell = QuantumLSTMCell(input_dim, hidden_dim, n_qubits)
        else:
            self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=False)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        if self.use_quantum:
            seq_len, batch, _ = embeds.shape
            hx = torch.zeros(batch, self.hidden_dim, device=embeds.device)
            cx = torch.zeros(batch, self.hidden_dim, device=embeds.device)
            outputs = []
            for t in range(seq_len):
                hx, cx = self.lstm_cell(embeds[t], hx, cx)
                outputs.append(hx.unsqueeze(0))
            outputs = torch.cat(outputs, dim=0)
        else:
            outputs, _ = self.lstm(embeds)
        tag_logits = self.hidden2tag(outputs)
        return F.log_softmax(tag_logits, dim=2)

__all__ = ["FullyConnectedQuantumLayer", "QuantumEstimatorQNN",
           "QuantumGateCircuit", "QuantumLSTMCell", "UnifiedQLSTM"]
