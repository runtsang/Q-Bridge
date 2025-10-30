import torch
from torch import nn
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Parameter
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Sampler
from qiskit_machine_learning.neural_networks import SamplerQNN

class FraudDetectionHybrid(nn.Module):
    """
    Quantum‑augmented fraud‑detection model that replaces the classical photonic
    layers with a variational RealAmplitudes circuit executed on a quantum sampler.
    """
    def __init__(self, input_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.sampler = Sampler()
        self.circuit = self._build_circuit()
        # The circuit has no trainable weights; all parameters are input‑encoded.
        self.qnn = SamplerQNN(circuit=self.circuit,
                              weight_params=[],
                              interpret=lambda x: x,
                              output_shape=1,
                              sampler=self.sampler)

    def _build_circuit(self) -> QuantumCircuit:
        qr = QuantumRegister(self.input_dim)
        qc = QuantumCircuit(qr)
        # Encode each input feature with an RY rotation
        for i in range(self.input_dim):
            qc.ry(Parameter(f'x{i}'), qr[i])
        qc.barrier()
        # Entangling layer
        for i in range(self.input_dim - 1):
            qc.cx(qr[i], qr[i + 1])
        qc.barrier()
        qc.measure_all()
        return qc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, input_dim)
        return self.qnn(x)

__all__ = ["FraudDetectionHybrid"]
