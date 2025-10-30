"""Hybrid self‑attention implemented as a Qiskit variational circuit.

The class builds a parameterised quantum circuit that applies rotations
and controlled‑rotations to encode the input.  The output distribution
over measurement outcomes is interpreted as attention weights, and a
SamplerQNN is used to produce the probability distribution, mirroring
the classical sampler network."""
import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import ParameterVector
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit.primitives import StatevectorSampler

class HybridSelfAttention:
    """
    Quantum self‑attention block that outputs a probability distribution
    over sequence positions via a parameterised circuit and a SamplerQNN.
    """
    def __init__(self, n_qubits: int, seq_len: int):
        self.n_qubits = n_qubits
        self.seq_len = seq_len
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")
        self.backend = qiskit.Aer.get_backend("qasm_simulator")
        self._build_circuit()

    def _build_circuit(self):
        self.circuit = QuantumCircuit(self.qr, self.cr)

        # Rotation parameters per qubit
        self.rotation_params = ParameterVector("rot", 3 * self.n_qubits)
        # Entanglement parameters between adjacent qubits
        self.entangle_params = ParameterVector("ent", self.n_qubits - 1)

        for i in range(self.n_qubits):
            self.circuit.rx(self.rotation_params[3 * i], i)
            self.circuit.ry(self.rotation_params[3 * i + 1], i)
            self.circuit.rz(self.rotation_params[3 * i + 2], i)

        for i in range(self.n_qubits - 1):
            self.circuit.crx(self.entangle_params[i], i, i + 1)

        self.circuit.measure(self.qr, self.cr)

        # Wrap as a SamplerQNN to produce probability distribution
        self.sampler_qnn = SamplerQNN(
            circuit=self.circuit,
            input_params=ParameterVector("input", self.seq_len),
            weight_params=ParameterVector("weight", 4),
            sampler=StatevectorSampler()
        )

    def run(self, rotation_vals: np.ndarray, entangle_vals: np.ndarray,
            input_vals: np.ndarray, shots: int = 1024) -> np.ndarray:
        """
        Execute the circuit and return a probability distribution over
        measurement outcomes that is interpreted as attention weights.
        """
        param_dict = {}
        for i, val in enumerate(rotation_vals):
            param_dict[self.rotation_params[i]] = val
        for i, val in enumerate(entangle_vals):
            param_dict[self.entangle_params[i]] = val

        for i, val in enumerate(input_vals):
            param_dict[self.sampler_qnn.input_params[i]] = val

        bound_circuit = self.circuit.bind_parameters(param_dict)
        job = qiskit.execute(bound_circuit, self.backend, shots=shots)
        result = job.result()
        counts = result.get_counts(bound_circuit)

        probs = np.zeros(self.seq_len)
        for bitstring, cnt in counts.items():
            idx = int(bitstring, 2) % self.seq_len
            probs[idx] += cnt
        probs = probs / probs.sum()
        return probs

__all__ = ["HybridSelfAttention"]
