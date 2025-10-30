import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.random import random_circuit
from qiskit.primitives import Sampler
from qiskit_machine_learning.neural_networks import SamplerQNN as QSamplerQNN
import torch
import torch.nn as nn
import torchquantum as tq

class QuantumRegression(tq.QuantumModule):
    """
    Simple quantum regression head based on a random layer and parameterized rotations.
    """
    def __init__(self, num_wires: int = 4):
        super().__init__()
        self.num_wires = num_wires
        self.random_layer = tq.RandomLayer(n_ops=20, wires=list(range(num_wires)))
        self.rx = tq.RX(has_params=True, trainable=True)
        self.ry = tq.RY(has_params=True, trainable=True)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(num_wires, 1)

    def forward(self, qdev: tq.QuantumDevice) -> torch.Tensor:
        self.random_layer(qdev)
        for wire in range(self.num_wires):
            self.rx(qdev, wires=wire)
            self.ry(qdev, wires=wire)
        features = self.measure(qdev)
        return self.head(features).squeeze(-1)

class Gen375:
    """
    Quantum‑centric implementation of the hybrid model.
    The architecture mirrors the classical `Gen375` but replaces the convolution,
    self‑attention, and sampler sub‑modules with Qiskit circuits.  The regression
    head is a small quantum module built with TorchQuantum.
    """
    def __init__(self,
                 kernel_size: int = 2,
                 threshold: float = 0.0,
                 n_qubits: int = 4,
                 num_wires: int = 4):
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.n_qubits = n_qubits
        self.num_wires = num_wires

        # Convolution circuit
        self.conv_qc = self._build_conv_circuit()
        # Attention circuit
        self.attn_qc = self._build_attn_circuit()
        # Sampler QNN
        self.sampler_qc = self._build_sampler_circuit()
        self.sampler = QSamplerQNN(
            circuit=self.sampler_qc,
            input_params=[qiskit.circuit.Parameter(f"inp{i}") for i in range(n_qubits)],
            weight_params=[qiskit.circuit.Parameter(f"w{i}") for i in range(2)],
            sampler=Sampler()
        )
        # Quantum regression head
        self.regressor = QuantumRegression(num_wires=num_wires)

        # Backend for execution
        self.backend = qiskit.Aer.get_backend("qasm_simulator")

    def _build_conv_circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.n_qubits)
        theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            qc.rx(theta[i], i)
        qc += random_circuit(self.n_qubits, 2)
        qc.measure_all()
        return qc

    def _build_attn_circuit(self) -> QuantumCircuit:
        qr = QuantumRegister(self.n_qubits, "q")
        cr = ClassicalRegister(self.n_qubits, "c")
        qc = QuantumCircuit(qr, cr)
        rot = [qiskit.circuit.Parameter(f"rot{i}") for i in range(3 * self.n_qubits)]
        ent = [qiskit.circuit.Parameter(f"ent{i}") for i in range(self.n_qubits - 1)]
        for i in range(self.n_qubits):
            qc.rx(rot[3 * i], i)
            qc.ry(rot[3 * i + 1], i)
            qc.rz(rot[3 * i + 2], i)
        for i in range(self.n_qubits - 1):
            qc.crx(ent[i], i, i + 1)
        qc.measure_all()
        return qc

    def _build_sampler_circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.n_qubits)
        inputs = [qiskit.circuit.Parameter(f"inp{i}") for i in range(self.n_qubits)]
        weights = [qiskit.circuit.Parameter(f"w{i}") for i in range(2)]
        for i in range(self.n_qubits):
            qc.ry(inputs[i], i)
        qc.cx(0, 1)
        qc.ry(weights[0], 0)
        qc.ry(weights[1], 1)
        qc.measure_all()
        return qc

    def _run_circuit(self, qc: QuantumCircuit, params: dict, shots: int = 1024) -> float:
        job = qiskit.execute(qc, self.backend, shots=shots, parameter_binds=[params])
        result = job.result()
        counts = result.get_counts(qc)
        total = shots * self.n_qubits
        ones = sum(sum(int(bit) for bit in key) * val for key, val in counts.items())
        return ones / total

    def forward(self, data: np.ndarray) -> tuple[torch.Tensor, np.ndarray]:
        """
        Parameters
        ----------
        data
            2‑D array of shape (kernel_size, kernel_size) with values in [0, 255].

        Returns
        -------
        out
            Regression output as a 0‑d tensor.
        probs
            Sampler probabilities as a 1‑D NumPy array of length 2.
        """
        flat = data.flatten()
        conv_params = {f"theta{i}": np.pi if v > self.threshold else 0 for i, v in enumerate(flat)}
        conv_out = self._run_circuit(self.conv_qc, conv_params)

        rot_params = {f"rot{i}": np.random.uniform(0, 2 * np.pi) for i in range(3 * self.n_qubits)}
        ent_params = {f"ent{i}": np.random.uniform(0, 2 * np.pi) for i in range(self.n_qubits - 1)}
        attn_out = self._run_circuit(self.attn_qc, {**rot_params, **ent_params})

        sampler_params = {f"inp{i}": conv_out for i in range(self.n_qubits)}
        sampler_counts = self.sampler.run(sampler_params).values()
        probs = np.array([sampler_counts.get('0'*self.n_qubits, 0),
                          sampler_counts.get('1'*self.n_qubits, 0)]) / sum(sampler_counts.values())

        # Encode features into a quantum device for regression
        qdev = tq.QuantumDevice(n_wires=self.num_wires, bsz=1, device="cpu")
        qdev.rY(qdev, 0, conv_out)
        qdev.rY(qdev, 1, attn_out)
        out = self.regressor(qdev).item()

        return out, probs

__all__ = ["Gen375"]
