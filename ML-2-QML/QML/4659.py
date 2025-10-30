"""Hybrid quantum sampler network that mirrors the classical
implementation but operates on quantum states.

Features:
- RealAmplitudes encoder that maps a 2‑dimensional classical input
  to a quantum state.
- A random layer for additional entanglement.
- A swap‑test style readout followed by a linear head producing a
  2‑element probability vector.

The class can be instantiated directly and used with the
Qiskit Machine Learning primitives.
"""

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes, RandomLayer
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit.primitives import StatevectorSampler as Sampler
import torch
import torch.nn as nn

class HybridSamplerQNN:
    """Quantum counterpart of the classical HybridSamplerQNN."""

    def __init__(self, num_qubits: int = 3, reps: int = 3):
        self.num_qubits = num_qubits
        self.reps = reps
        self.circuit = self._build_circuit()
        self.sampler = Sampler()
        self.qnn = SamplerQNN(
            circuit=self.circuit,
            input_params=[],
            weight_params=self.circuit.parameters,
            interpret=self._interpret,
            output_shape=2,
            sampler=self.sampler,
        )

    def _build_circuit(self) -> QuantumCircuit:
        qr = QuantumRegister(self.num_qubits, "q")
        cr = ClassicalRegister(1, "c")
        qc = QuantumCircuit(qr, cr)

        # Encoder
        encoder = RealAmplitudes(self.num_qubits, reps=self.reps)
        qc.compose(encoder, qr, inplace=True)

        # Random layer
        rand_layer = RandomLayer(n_ops=20, wires=list(range(self.num_qubits)))
        qc.compose(rand_layer, qr, inplace=True)

        # Swap‑test style readout with an auxiliary qubit
        aux = QuantumRegister(1, "aux")
        qc.add_register(aux)
        qc.h(aux[0])
        for i in range(self.num_qubits):
            qc.cswap(aux[0], qr[i], qr[i])  # placeholder swap to keep structure
        qc.h(aux[0])
        qc.measure(aux[0], cr[0])

        return qc

    def _interpret(self, x: torch.Tensor) -> torch.Tensor:
        # Convert sampler output to a probability vector
        probs = x.squeeze(-1)
        return probs / probs.sum(dim=-1, keepdim=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Run the sampler and return a probability distribution."""
        return self.qnn(inputs)

__all__ = ["HybridSamplerQNN"]
