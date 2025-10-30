"""Hybrid quantum sampler.

The quantum component accepts the weight parameters produced by the
classical encoder and constructs a parameterized circuit that
generates sampling probabilities over the computational basis.
The circuit consists of input rotations, entangling gates, a
parameterised weight layer, and a final entangling stage."""
from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import ParameterVector

class HybridSamplerQNN:
    """
    Quantum component of the hybrid sampler.

    Attributes
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    input_params : ParameterVector
        Parameters for the input rotation gates.
    weight_params : ParameterVector
        Parameters for the weight rotation gates.
    circuit : QuantumCircuit
        Parameterised circuit template.
    """
    def __init__(self, num_qubits: int = 2) -> None:
        self.num_qubits = num_qubits
        self.input_params = ParameterVector("input", num_qubits)
        self.weight_params = ParameterVector("weight", 4)
        self.circuit = self._build_circuit()

    def _build_circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.num_qubits)
        # Input rotations
        for i in range(self.num_qubits):
            qc.ry(self.input_params[i], i)
        # Entanglement
        for i in range(self.num_qubits - 1):
            qc.cx(i, i + 1)
        # Weight rotations
        for i in range(4):
            target = i % self.num_qubits
            qc.ry(self.weight_params[i], target)
        # Final entanglement
        for i in range(self.num_qubits - 1):
            qc.cx(i, i + 1)
        return qc

    def sample(self, input_vals: list[float], weight_vals: list[float],
               shots: int = 1024) -> np.ndarray:
        """
        Sample from the quantum circuit.

        Parameters
        ----------
        input_vals : list[float]
            Numerical values for the input rotation parameters.
        weight_vals : list[float]
            Numerical values for the weight rotation parameters.
        shots : int
            Number of shots for the QASM simulator.

        Returns
        -------
        probs : np.ndarray
            Probability distribution over the ``2**num_qubits`` basis
            states.
        """
        bound_params = {p: v for p, v in zip(self.input_params, input_vals)}
        bound_params.update({p: v for p, v in zip(self.weight_params, weight_vals)})
        bound_qc = self.circuit.bind_parameters(bound_params)

        backend = Aer.get_backend("qasm_simulator")
        job = execute(bound_qc, backend=backend, shots=shots)
        result = job.result()
        counts = result.get_counts()
        probs = np.zeros(2 ** self.num_qubits)
        for bitstring, count in counts.items():
            idx = int(bitstring, 2)
            probs[idx] = count / shots
        return probs

__all__ = ["HybridSamplerQNN"]
