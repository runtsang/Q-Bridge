import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.providers.aer import AerSimulator
from qiskit import transpile, assemble

class HybridQuantumSelfAttention:
    """
    Quantum self‑attention unit that mirrors the classical interface.
    The circuit first encodes the input token with a ZFeatureMap,
    applies parameterised rotations (rotation_params) on each qubit,
    then entangles neighboring qubits using controlled‑RZ gates
    (entangle_params).  The measurement of Z on each qubit yields
    a probability vector that is interpreted as attention scores.
    """

    def __init__(self,
                 n_qubits: int,
                 backend=None,
                 shots: int = 1024):
        self.n_qubits = n_qubits
        self.backend = backend or AerSimulator()
        self.shots = shots

        # Parameter vectors
        self.rotation_params = ParameterVector("theta_r", length=n_qubits * 3)
        self.entangle_params = ParameterVector("theta_e", length=n_qubits - 1)

        # Feature map
        self.feature_map = ZFeatureMap(n_qubits, reps=1, entanglement='linear')

        # Build the base circuit
        self.circuit = QuantumCircuit(n_qubits, name="self_attention")
        self.circuit.append(self.feature_map.to_instruction(), range(n_qubits))

        # Rotation gates
        for i in range(n_qubits):
            self.circuit.rx(self.rotation_params[3 * i], i)
            self.circuit.ry(self.rotation_params[3 * i + 1], i)
            self.circuit.rz(self.rotation_params[3 * i + 2], i)

        # Entanglement
        for i in range(n_qubits - 1):
            self.circuit.crx(self.entangle_params[i], i, i + 1)

        # Measurement of Z (via Pauli Z expectation)
        self.circuit.measure_all()

    def run(self,
            rotation_params: np.ndarray,
            entangle_params: np.ndarray,
            shots: int = None) -> np.ndarray:
        """
        Execute the circuit with supplied parameters and return the
        expectation value of Z on each qubit (attention scores).

        Parameters
        ----------
        rotation_params
            Array of shape (n_qubits * 3,)
        entangle_params
            Array of shape (n_qubits - 1,)
        shots
            Number of shots for the backend.  If None, defaults to self.shots.

        Returns
        -------
        np.ndarray
            Shape (n_qubits,) – normalized attention scores derived
            from the measurement distribution.
        """
        shots = shots or self.shots

        # Bind parameters
        param_bindings = [
            {self.rotation_params[i]: rotation_params[i]
             for i in range(len(self.rotation_params))},
            {self.entangle_params[i]: entangle_params[i]
             for i in range(len(self.entangle_params))}
        ]

        # Compile and execute
        compiled = transpile(self.circuit, self.backend)
        qobj = assemble(compiled,
                        shots=shots,
                        parameter_binds=[param_bindings[0], param_bindings[1]])

        job = self.backend.run(qobj)
        result = job.result()

        # Compute expectation values of Z for each qubit
        counts = result.get_counts(self.circuit)
        exp_z = np.zeros(self.n_qubits)
        for state, cnt in counts.items():
            bits = np.array([int(b) for b in state[::-1]])  # reverse due to ordering
            prob = cnt / shots
            # Map |0> -> +1, |1> -> -1
            exp_z += (1 - 2 * bits) * prob

        # Normalize to [0,1] as attention weights
        scores = (exp_z + 1) / 2
        return scores / scores.sum()

__all__ = ["HybridQuantumSelfAttention"]
