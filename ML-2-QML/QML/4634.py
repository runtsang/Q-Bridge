import numpy as np
import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
import torch
from torch import nn
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit.primitives import StatevectorSampler

class FCL:
    """
    Quantum‑enhanced fully‑connected layer that uses a parameterised quantum
    circuit for feature encoding, a variational ansatz, and a classical
    classifier head.  The implementation is inspired by the
    QuantumClassifierModel and SamplerQNN reference pairs.
    """

    def __init__(self, num_qubits: int = 2, depth: int = 2, shots: int = 100):
        """Construct the quantum circuit, sampler, and classical post‑processing.

        Args:
            num_qubits: number of qubits / input dimensions.
            depth: number of variational layers in the ansatz.
            shots: number of shots for the backend simulator.
        """
        self.num_qubits = num_qubits
        self.depth = depth
        self.shots = shots
        self.backend = qiskit.Aer.get_backend("qasm_simulator")

        self.circuit, self.encoding_params, self.weight_params, self.observables = self._build_circuit()

        self.sampler = StatevectorSampler(self.backend)
        self.sampler_qnn = SamplerQNN(
            circuit=self.circuit,
            input_params=self.encoding_params,
            weight_params=self.weight_params,
            sampler=self.sampler,
        )

        # Classical classifier mapping expectation values to logits
        self.classifier = nn.Linear(num_qubits, 2)

    def _build_circuit(self):
        encoding = ParameterVector("x", self.num_qubits)
        weights = ParameterVector("theta", self.num_qubits * self.depth)

        qc = QuantumCircuit(self.num_qubits)

        # Data‑encoding layer
        for param, qubit in zip(encoding, range(self.num_qubits)):
            qc.rx(param, qubit)

        # Variational layers
        idx = 0
        for _ in range(self.depth):
            for qubit in range(self.num_qubits):
                qc.ry(weights[idx], qubit)
                idx += 1
            for qubit in range(self.num_qubits - 1):
                qc.cz(qubit, qubit + 1)

        # Observables: Z on each qubit
        observables = [
            SparsePauliOp("I" * i + "Z" + "I" * (self.num_qubits - i - 1))
            for i in range(self.num_qubits)
        ]
        return qc, list(encoding), list(weights), observables

    def run(self, inputs: np.ndarray) -> np.ndarray:
        """
        Execute the quantum circuit for each input vector, compute expectation
        values of the Z observables, and classify with the classical head.

        Args:
            inputs: (batch_size, num_qubits) array of real values.

        Returns:
            probabilities: (batch_size, 2) array of softmax probabilities.
        """
        probs = []
        for input_vector in inputs:
            # Bind the encoding parameters
            bind_dict = {p: val for p, val in zip(self.encoding_params, input_vector)}

            # Run the sampler (weight parameters are sampled by the SamplerQNN)
            sample_counts = self.sampler_qnn.run(params=bind_dict)

            # Convert counts to probabilities
            counts = {state: prob for state, prob in sample_counts.items()}
            # Compute expectation values for each Z observable
            expectation = np.zeros(self.num_qubits)
            for i, _ in enumerate(self.observables):
                exp_val = 0.0
                for state, prob in counts.items():
                    # Qiskit returns bitstrings with little‑endian convention
                    bit = int(state[::-1][i])  # reverse to match qubit ordering
                    exp_val += (1 if bit == 0 else -1) * prob
                expectation[i] = exp_val

            # Feed into classical classifier
            logits = self.classifier(torch.tensor(expectation, dtype=torch.float32))
            probs.append(torch.softmax(logits, dim=-1).detach().numpy())
        return np.array(probs)

__all__ = ["FCL"]
