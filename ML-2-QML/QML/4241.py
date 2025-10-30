import torch
import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.neural_networks import SamplerQNN

class HybridAutoencoder:
    """Quantum autoencoder circuit using a parameterized ansatz and a sampler QNN."""
    def __init__(self, input_dim: int, latent_dim: int, reps: int = 3):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.reps = reps

        self.sampler = StatevectorSampler()
        self.input_params = ParameterVector("x", input_dim)
        self.circuit = self._build_circuit()

        weight_params = [p for p in self.circuit.parameters if p not in self.input_params]
        self.qnn = SamplerQNN(
            circuit=self.circuit,
            input_params=self.input_params,
            weight_params=weight_params,
            interpret=lambda x: x,
            output_shape=latent_dim,
            sampler=self.sampler,
        )

    def _build_circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.latent_dim)
        # Encode classical data via RX rotations
        for i, param in enumerate(self.input_params):
            qc.rx(param, i % self.latent_dim)
        # Variational ansatz
        qc.append(RealAmplitudes(self.latent_dim, reps=self.reps), range(self.latent_dim))
        return qc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Execute the sampler QNN on the input tensor."""
        return self.qnn(x)
