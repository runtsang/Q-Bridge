import numpy as np
import torch
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.primitives import Sampler
from qiskit_machine_learning.neural_networks import SamplerQNN as QiskitSamplerQNN

class SamplerQNN:
    """
    Quantum sampler network that generalises the original 2‑qubit circuit.
    It employs a 2‑qubit variational circuit with three layers of entangling Ry rotations.
    The network outputs a probability distribution over 4 possible 2‑bit outcomes,
    which can be queried or sampled via a PyTorch interface.
    """
    def __init__(self, num_shots: int = 1024) -> None:
        # Parameter vectors for input and variational weights
        self.input_params = ParameterVector("x", 2)
        self.weight_params = ParameterVector("w", 6)  # 3 layers × 2 qubits

        # Build the circuit
        self.circuit = QuantumCircuit(2, 2)
        # Input rotations
        self.circuit.ry(self.input_params[0], 0)
        self.circuit.ry(self.input_params[1], 1)
        self.circuit.cx(0, 1)
        # Variational layers
        for layer in range(3):
            w0 = self.weight_params[layer * 2]
            w1 = self.weight_params[layer * 2 + 1]
            self.circuit.ry(w0, 0)
            self.circuit.ry(w1, 1)
            self.circuit.cx(0, 1)
        # Measurement
        self.circuit.measure_all()

        # Sampler primitive
        self.sampler = Sampler()
        self.qiskit_sampler = QiskitSamplerQNN(
            circuit=self.circuit,
            input_params=self.input_params,
            weight_params=self.weight_params,
            sampler=self.sampler,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Compute the probability distribution for a batch of 2‑dimensional inputs.
        Args:
            inputs: Tensor of shape (batch, 2) with values in the real domain.
        Returns:
            probs: Tensor of shape (batch, 4) with probabilities for each 2‑bit outcome.
        """
        batch_size = inputs.shape[0]
        probs = torch.zeros(batch_size, 4, dtype=torch.float32)
        for i in range(batch_size):
            param_dict = {p: float(v) for p, v in zip(self.input_params, inputs[i].tolist())}
            result = self.qiskit_sampler.run(param_dict)
            # Convert dict to ordered probability array
            probs[i] = torch.tensor(
                [result.get(f"{b:02b}", 0.0) for b in range(4)],
                dtype=torch.float32,
            )
        return probs

    def sample(self, inputs: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        """
        Sample from the output distribution for each input in the batch.
        """
        probs = self.forward(inputs)
        samples = torch.multinomial(probs, num_samples, replacement=True)
        return samples

__all__ = ["SamplerQNN"]
