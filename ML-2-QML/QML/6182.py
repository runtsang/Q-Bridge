"""Quantum self‑attention head based on a parameterised SamplerQNN circuit.

The circuit encodes a query–key pair as two rotation angles on a 2‑qubit
system and uses a small set of trainable weights to bias the measurement
outcome.  The probability of outcome '1' is interpreted as the attention
weight for that pair.  The class is deliberately lightweight so that it
can be called from a classical wrapper without any deep‑learning
framework dependencies.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit.primitives import StatevectorSampler
from qiskit.providers.aer import AerSimulator

class HybridSelfAttention:
    """Quantum attention head used by the classical hybrid implementation."""
    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        # Build a reusable 2‑qubit SamplerQNN circuit
        self._build_sampler_circuit()
        self.backend = AerSimulator()
        self.sampler = StatevectorSampler(self.backend)

    def _build_sampler_circuit(self):
        """Create the 2‑qubit SamplerQNN circuit with trainable weights."""
        self.input_params = ParameterVector("input", 2)
        self.weight_params = ParameterVector("weight", 4)

        qc = QuantumCircuit(2)
        qc.ry(self.input_params[0], 0)
        qc.ry(self.input_params[1], 1)
        qc.cx(0, 1)
        qc.ry(self.weight_params[0], 0)
        qc.ry(self.weight_params[1], 1)
        qc.cx(0, 1)
        qc.ry(self.weight_params[2], 0)
        qc.ry(self.weight_params[3], 1)
        # No measurement – we use statevector sampler
        self.circuit = qc

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
        shots: int = 1024,
    ) -> np.ndarray:
        """
        Compute attention logits for all query‑key pairs.

        Parameters
        ----------
        rotation_params : np.ndarray
            Length 3*n_qubits array used to set the weight parameters.
        entangle_params : np.ndarray
            Length n_qubits‑1 array; unused in this toy implementation.
        inputs : np.ndarray
            Shape (N, 2) where each row contains a query and a key value.
        shots : int, optional
            Number of shots; not required when using statevector sampler.

        Returns
        -------
        np.ndarray
            Shape (N, 2) – probabilities of measurement outcomes '0' and '1'.
        """
        # Map the first four rotation angles to the SamplerQNN weights
        weight_vals = rotation_params[:4] if rotation_params.size >= 4 else np.zeros(4)
        probs = []
        for pair in inputs:
            # Bind weight parameters to the circuit
            bound_circuit = self.circuit.bind_parameters(
                {
                    self.weight_params[0]: weight_vals[0],
                    self.weight_params[1]: weight_vals[1],
                    self.weight_params[2]: weight_vals[2],
                    self.weight_params[3]: weight_vals[3],
                }
            )
            # Create a new SamplerQNN instance for this bound circuit
            sampler_qnn = SamplerQNN(
                circuit=bound_circuit,
                input_params=self.input_params,
                weight_params=self.weight_params,
                sampler=self.sampler,
            )
            outcome_probs = sampler_qnn.evaluate(pair.reshape(1, -1), shots=shots)
            probs.append([outcome_probs.get('0', 0.0), outcome_probs.get('1', 0.0)])
        return np.array(probs)

__all__ = ["HybridSelfAttention"]
