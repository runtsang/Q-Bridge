import numpy as np
import torch
import torchquantum as tq
import torch.nn as nn
from qiskit import QuantumCircuit, Aer, execute, transpile, assemble
from qiskit.circuit import ParameterVector
from qiskit.primitives import Sampler as StatevectorSampler

class QuantumHybridLayer(tq.QuantumModule):
    """Variational quantum circuit used as a feature extractor."""

    def __init__(self, num_wires: int, n_random_ops: int = 30):
        super().__init__()
        self.num_wires = num_wires
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        self.random_layer = tq.RandomLayer(n_ops=n_random_ops, wires=list(range(num_wires)))
        self.rx = tq.RX(has_params=True, trainable=True)
        self.ry = tq.RY(has_params=True, trainable=True)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(num_wires, num_wires)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.num_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        self.random_layer(qdev)
        for w in range(self.num_wires):
            self.rx(qdev, wires=w)
            self.ry(qdev, wires=w)
        features = self.measure(qdev)
        return self.head(features).squeeze(-1)

class SamplerQNN:
    """Parameterised circuit that samples a 2‑qubit state using Qiskit."""

    def __init__(self):
        self.inputs = ParameterVector("input", 2)
        self.weights = ParameterVector("weight", 4)

        self.circuit = QuantumCircuit(2)
        self.circuit.ry(self.inputs[0], 0)
        self.circuit.ry(self.inputs[1], 1)
        self.circuit.cx(0, 1)
        self.circuit.ry(self.weights[0], 0)
        self.circuit.ry(self.weights[1], 1)
        self.circuit.cx(0, 1)
        self.circuit.ry(self.weights[2], 0)
        self.circuit.ry(self.weights[3], 1)

        self.backend = Aer.get_backend("statevector_simulator")
        self.sampler = StatevectorSampler(self.backend)

    def sample(self, input_vals: np.ndarray) -> np.ndarray:
        """Return sampled statevector for given input parameters."""
        param_dict = {self.inputs[i]: val for i, val in enumerate(input_vals)}
        job = self.sampler.run(self.circuit, param_dict)
        result = job.result()
        return result.get_statevector()

__all__ = ["QuantumHybridLayer", "SamplerQNN"]
