from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import SamplerQNN as QSampler
from qiskit.primitives import StatevectorSampler
from qiskit.providers.aer import AerSimulator
import numpy as np

class SamplerQNN:
    """Quantum sampler mirroring the classical interface."""
    def __init__(self,
                 num_qubits: int,
                 depth: int,
                 shots: int = 1024,
                 backend: str = 'qasm_simulator',
                 threshold: float = 0.5) -> None:
        self.num_qubits = num_qubits
        self.depth = depth
        self.shots = shots
        self.backend = AerSimulator(name=backend)
        self.threshold = threshold
        self.circuit, self.encoding, self.weights, self.observables = self._build_circuit()

        # Wrap into Qiskit ML SamplerQNN for easy sampling
        self.sampler = StatevectorSampler()
        self.sampler_qnn = QSampler(circuit=self.circuit,
                                    input_params=self.encoding,
                                    weight_params=self.weights,
                                    sampler=self.sampler)

    def _build_circuit(self):
        """Construct a layered ansatz with encoding and variational parameters."""
        encoding = ParameterVector("x", self.num_qubits)
        weights = ParameterVector("theta", self.num_qubits * self.depth)

        qc = QuantumCircuit(self.num_qubits)
        for q, param in zip(range(self.num_qubits), encoding):
            qc.rx(param, q)

        idx = 0
        for _ in range(self.depth):
            for q in range(self.num_qubits):
                qc.ry(weights[idx], q)
                idx += 1
            for q in range(self.num_qubits - 1):
                qc.cz(q, q + 1)

        observables = [SparsePauliOp("I" * i + "Z" + "I" * (self.num_qubits - i - 1))
                       for i in range(self.num_qubits)]
        return qc, list(encoding), list(weights), observables

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Execute the circuit for a batch of classical inputs.

        Parameters
        ----------
        inputs : np.ndarray
            Shape (batch, num_qubits). Values should be in [0, 2π] for encoding.

        Returns
        -------
        np.ndarray
            Probabilities over two outcomes, shape (batch, 2).
        """
        # Bind encoding parameters
        param_binds = [{param: val for param, val in zip(self.encoding, inp)}
                       for inp in inputs]
        # Sample probabilities from the circuit
        probs = []
        for bind in param_binds:
            result = self.sampler_qnn.run(bind)
            # result is a dictionary mapping bitstrings to probabilities
            # Convert to two‑outcome probability: |0> vs |1>
            p0 = sum(p for bits, p in result.items() if bits.count('1') == 0)
            p1 = 1.0 - p0
            probs.append([p0, p1])
        return np.array(probs)

    def sample(self, inputs: np.ndarray, num_samples: int = 1) -> np.ndarray:
        """
        Draw quantum samples from the output distribution.

        Parameters
        ----------
        inputs : np.ndarray
            Input batch.
        num_samples : int
            Number of samples per batch element.

        Returns
        -------
        np.ndarray
            Sampled labels of shape (batch, num_samples).
        """
        probs = self.forward(inputs)
        return np.random.choice(2, size=(probs.shape[0], num_samples), p=probs)
