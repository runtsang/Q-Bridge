import numpy as np
import strawberryfields as sf
from strawberryfields.ops import Dgate, N
import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import ParameterVector
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit.primitives import StatevectorSampler
from typing import Tuple

class QuantumPhotonicCircuit:
    """
    Photonic variational circuit that maps a 2‑dimensional input vector
    to an expectation value of the photon number operator.
    """
    def __init__(self, n_modes: int = 2):
        self.n_modes = n_modes

    def run(self, inputs: np.ndarray) -> np.ndarray:
        prog = sf.Program(self.n_modes)
        with prog.context as q:
            for i in range(self.n_modes):
                Dgate(inputs[i]) | q[i]
        eng = sf.Engine("gaussian")
        result = eng.run(prog).state
        exp = result.expectation_value(N(0))
        return np.array([exp.real])

class QuantumLSTM(nn.Module):
    """
    Quantum‑enhanced LSTM cell where each gate is realised by a small
    parametrised quantum circuit.
    """
    class QGate(tq.QuantumModule):
        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            self.encoder = tq.GeneralEncoder(
                [
                    {"input_idx": [0], "func": "rx", "wires": [0]},
                    {"input_idx": [1], "func": "rx", "wires": [1]},
                ]
            )
            self.params = nn.ModuleList([tq.RX(has_params=True, trainable=True) for _ in range(n_wires)])
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=x.shape[0], device=x.device)
            self.encoder(qdev, x)
            for wire, gate in enumerate(self.params):
                gate(qdev, wires=wire)
            return self.measure(qdev)

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.forget = self.QGate(n_qubits)
        self.input = self.QGate(n_qubits)
        self.update = self.QGate(n_qubits)
        self.output = self.QGate(n_qubits)

        self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)

    def forward(self, inputs: torch.Tensor, states: Tuple[torch.Tensor, torch.Tensor] = None):
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

    def _init_states(self, inputs: torch.Tensor, states: Tuple[torch.Tensor, torch.Tensor] = None):
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return torch.zeros(batch_size, self.hidden_dim, device=device), torch.zeros(batch_size, self.hidden_dim, device=device)

class QuantumFullyConnected:
    """
    Parameterised single‑qubit circuit that evaluates a linear combination
    of input angles as a quantum expectation.
    """
    def __init__(self, n_qubits: int = 2):
        self.n_qubits = n_qubits
        self.backend = Aer.get_backend("qasm_simulator")
        self.shots = 1024
        self.theta = ParameterVector("theta", n_qubits)

    def run(self, inputs: np.ndarray) -> np.ndarray:
        qc = QuantumCircuit(self.n_qubits)
        for i in range(self.n_qubits):
            qc.ry(self.theta[i], i)
        qc.measure_all()
        job = execute(qc, self.backend, shots=self.shots,
                      parameter_binds=[{self.theta[i]: inputs[i]} for i in range(self.n_qubits)])
        result = job.result()
        counts = result.get_counts(qc)
        probs = {k: v / self.shots for k, v in counts.items()}
        expectation = sum(int(k, 2) * p for k, p in probs.items())
        return np.array([expectation])

class QuantumSampler:
    """
    Quantum sampler that maps a 2‑dimensional input to a probability
    vector via a parameterised circuit.
    """
    def __init__(self, n_qubits: int = 2):
        self.n_qubits = n_qubits
        self.input_params = ParameterVector("input", n_qubits)
        self.weight_params = ParameterVector("weight", 4)
        qc = QuantumCircuit(n_qubits)
        for i in range(n_qubits):
            qc.ry(self.input_params[i], i)
        qc.cx(0, 1)
        for i in range(4):
            qc.ry(self.weight_params[i], i % n_qubits)
        qc.cx(0, 1)
        sampler = StatevectorSampler()
        self.sampler_qnn = SamplerQNN(circuit=qc,
                                      input_params=self.input_params,
                                      weight_params=self.weight_params,
                                      sampler=sampler)

    def run(self, inputs: np.ndarray) -> np.ndarray:
        return self.sampler_qnn.run(inputs)

__all__ = ["QuantumPhotonicCircuit", "QuantumLSTM", "QuantumFullyConnected", "QuantumSampler"]
