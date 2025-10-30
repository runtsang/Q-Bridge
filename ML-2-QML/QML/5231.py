"""Quantum implementation of the hybrid convolutional filter."""
import numpy as np
import torch
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute
from qiskit.circuit.random import random_circuit
from QLSTM import QLSTM
from Autoencoder import Autoencoder

class QuantumSelfAttention:
    """Quantum self‑attention block based on a parametrised RX‑RY‑RZ circuit."""
    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")
        self.backend = qiskit.Aer.get_backend("qasm_simulator")
        self.shots = 1024

    def _build_circuit(self, rotation_params, entangle_params):
        circuit = QuantumCircuit(self.qr, self.cr)
        for i in range(self.n_qubits):
            circuit.rx(rotation_params[3 * i], i)
            circuit.ry(rotation_params[3 * i + 1], i)
            circuit.rz(rotation_params[3 * i + 2], i)
        for i in range(self.n_qubits - 1):
            circuit.crx(entangle_params[i], i, i + 1)
        circuit.measure(self.qr, self.cr)
        return circuit

    def run(self, rotation_params, entangle_params):
        circuit = self._build_circuit(rotation_params, entangle_params)
        job = execute(circuit, self.backend, shots=self.shots)
        result = job.result().get_counts(circuit)
        probs = np.zeros(self.n_qubits)
        for key, val in result.items():
            for idx, bit in enumerate(reversed(key)):
                if bit == '1':
                    probs[idx] += val
        probs /= self.shots
        return probs

class ConvGen114:
    """Hybrid quantum‑classical convolutional filter."""
    def __init__(self,
                 kernel_size: int = 2,
                 threshold: float = 0.0,
                 attention_type: str = 'quantum',
                 autoenc: bool = True,
                 lstm_qubits: int = 0):
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.attention_type = attention_type
        self.autoenc = autoenc
        self.lstm_qubits = lstm_qubits

        self.n_qubits = kernel_size ** 2
        self.backend = qiskit.Aer.get_backend("qasm_simulator")
        self.shots = 1024

        # Quantum convolution circuit
        self.circuit = QuantumCircuit(self.n_qubits)
        self.theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            self.circuit.rx(self.theta[i], i)
        self.circuit.barrier()
        self.circuit += random_circuit(self.n_qubits, 2)
        self.circuit.measure_all()

        # Auto‑encoder
        if autoenc:
            self.qautoencoder = Autoencoder()
        else:
            self.qautoencoder = None

        # Self‑attention
        if attention_type == 'quantum':
            self.attention = QuantumSelfAttention(self.n_qubits)
        else:
            self.attention = None

        # LSTM
        if lstm_qubits > 0:
            self.lstm = QLSTM(
                input_dim=self.n_qubits,
                hidden_dim=16,
                n_qubits=lstm_qubits,
            )
        else:
            self.lstm = None

    def run(self, data: np.ndarray | torch.Tensor) -> float:
        """Apply the quantum hybrid pipeline to a 2‑D input."""
        data_vec = np.reshape(data, (self.n_qubits,))
        # Parameter binding
        param_binds = []
        for val in data_vec:
            bind = {self.theta[i]: np.pi if val > self.threshold else 0
                    for i in range(self.n_qubits)}
            param_binds.append(bind)

        job = execute(self.circuit,
                      self.backend,
                      shots=self.shots,
                      parameter_binds=param_binds)
        result = job.result().get_counts(self.circuit)

        # Compute average |1> probability
        counts = 0
        for key, val in result.items():
            ones = sum(int(b) for b in key)
            counts += ones * val
        conv_out = counts / (self.shots * self.n_qubits)

        # Auto‑encoder
        if self.qautoencoder:
            ae_out = self.qautoencoder.forward()
            conv_out = conv_out * ae_out[0] + (1 - conv_out) * ae_out[1]

        # Self‑attention
        if self.attention:
            rot = np.random.rand(3 * self.n_qubits)
            ent = np.random.rand(self.n_qubits - 1)
            attn = self.attention.run(rot, ent)
            conv_out = conv_out * attn.mean()

        # LSTM
        if self.lstm:
            seq = torch.tensor([conv_out], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            lstm_out, _ = self.lstm(seq)
            output = lstm_out.mean().item()
        else:
            output = conv_out

        return output

def Conv() -> ConvGen114:
    """Convenience factory matching the original API."""
    return ConvGen114()

__all__ = ["ConvGen114", "Conv"]
