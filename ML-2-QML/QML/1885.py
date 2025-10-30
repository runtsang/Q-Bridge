from qiskit import QuantumCircuit, Aer
from qiskit.circuit import ParameterVector
from qiskit.primitives import Sampler
from qiskit_machine_learning.neural_networks import SamplerQNN as QiskitSamplerQNN
import numpy as np

class SamplerQNN:
    """
    Quantum sampler network based on a 2‑qubit variational circuit.
    The circuit contains ``layers`` entangling blocks with full sets of
    RX, RY and RZ rotations.  It can be executed on any Qiskit backend
    (default ``aer_simulator``) and wrapped by Qiskit‑ML’s SamplerQNN
    for compatibility with the classical API.
    """
    def __init__(self,
                 qubits: int = 2,
                 layers: int = 3,
                 backend_name: str = "aer_simulator",
                 seed: int | None = None) -> None:
        self.qubits = qubits
        self.layers = layers
        self.backend = Aer.get_backend(backend_name)
        self.seed = seed
        self.circuit = self._build_circuit()
        self.sampler = Sampler(backend=self.backend)
        # Parameters for the circuit
        self.input_params = ParameterVector("x", qubits)
        self.weight_params = ParameterVector("w", layers * qubits * 3)
        # Wrap with Qiskit‑ML SamplerQNN for API parity
        self.wrapper = QiskitSamplerQNN(circuit=self.circuit,
                                        input_params=self.input_params,
                                        weight_params=self.weight_params,
                                        sampler=self.sampler)

    def _build_circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.qubits)
        for _ in range(self.layers):
            for q in range(self.qubits):
                qc.ry(ParameterVector("w", 1)[0], q)  # placeholder, will be overridden
            # Entangling pattern
            for q in range(self.qubits - 1):
                qc.cx(q, q + 1)
            for q in range(1, self.qubits):
                qc.cx(q, q - 1)
        return qc

    def probabilities(self, inputs: np.ndarray) -> np.ndarray:
        """
        Evaluate the circuit for a batch of classical inputs and return
        the probability distribution over the computational basis.
        """
        if inputs.ndim == 1:
            inputs = inputs.reshape(1, -1)
        probs_list = []
        for x in inputs:
            bound = {p: v for p, v in zip(self.input_params, x)}
            result = self.sampler.run([self.circuit]).bind_parameters(bound).result()
            counts = result.get_counts()
            probs = np.zeros(2 ** self.qubits)
            for bitstring, count in counts.items():
                idx = int(bitstring[::-1], 2)
                probs[idx] = count / result.get_numshots()
            probs_list.append(probs)
        return np.array(probs_list)

    def sample(self, inputs: np.ndarray, n_shots: int = 1024) -> np.ndarray:
        """
        Draw samples from the quantum state prepared by the circuit.
        """
        probs = self.probabilities(inputs)
        samples = [np.random.choice(2 ** self.qubits, size=n_shots, p=p)
                   for p in probs]
        return np.array(samples)

__all__ = ["SamplerQNN"]
