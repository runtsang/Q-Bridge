import numpy as np
import qiskit
from qiskit import QuantumCircuit, execute, Aer
from qiskit.circuit import Parameter
from typing import Iterable

class FCL:
    """
    Variational quantum circuit mimicking a fully‑connected layer.
    The circuit is built from a stack of parameterised RY rotations
    followed by a configurable entanglement pattern.  The ``run`` method
    accepts a list of angles (``thetas``) and returns the expectation
    value of the Z observable on the first qubit.
    """
    def __init__(
        self,
        n_qubits: int = 1,
        layers: int = 2,
        backend=None,
        shots: int = 1024,
    ) -> None:
        self.n_qubits = n_qubits
        self.layers = layers
        self.shots = shots
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self._theta_params = [
            Parameter(f"θ_{l}_{q}") for l in range(layers) for q in range(n_qubits)
        ]
        self.circuit = self._build_circuit()

    def _build_circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.n_qubits)
        # Initial Hadamards
        qc.h(range(self.n_qubits))
        # Parameterised layers
        for l in range(self.layers):
            for q in range(self.n_qubits):
                qc.ry(self._theta_params[l * self.n_qubits + q], q)
            # Entanglement: a linear chain of CX gates
            for q in range(self.n_qubits - 1):
                qc.cx(q, q + 1)
        qc.measure_all()
        return qc

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Execute the circuit with the supplied parameters.

        Parameters
        ----------
        thetas : Iterable[float]
            Iterable of rotation angles.  The length must equal
            ``layers * n_qubits``; otherwise a ``ValueError`` is raised.

        Returns
        -------
        np.ndarray
            1‑D array containing the expectation value of Z on the first
            qubit (computed from measurement counts).
        """
        thetas = list(thetas)
        if len(thetas)!= len(self._theta_params):
            raise ValueError(
                f"Expected {len(self._theta_params)} angles, got {len(thetas)}."
            )
        param_bind = [{p: a for p, a in zip(self._theta_params, thetas)}]
        job = execute(
            self.circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=param_bind,
        )
        result = job.result()
        counts = result.get_counts(self.circuit)
        # Convert bitstrings to integers and compute expectation of Z
        exp_val = 0.0
        for bitstring, cnt in counts.items():
            # Convert bitstring to integer; bit 0 is the first qubit
            val = 1 if bitstring[-1] == "0" else -1  # Z eigenvalue
            exp_val += val * cnt
        exp_val /= self.shots
        return np.array([exp_val])

__all__ = ["FCL"]
