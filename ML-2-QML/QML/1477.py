"""Quantum variational circuit with multi‑qubit entanglement and parameter‑shift gradient."""
import numpy as np
import qiskit
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, execute, Aer
from qiskit.circuit import ParameterVector


class QuantumCircuitLayer:
    """
    Parameterized quantum circuit designed to emulate a fully‑connected layer.

    Parameters
    ----------
    n_qubits : int
        Number of qubits in the circuit.
    depth : int, default 1
        Number of layers of parameterized rotations and entangling gates.
    shots : int, default 1024
        Number of shots for the backend simulation.

    The circuit is built as:
        For each depth layer:
            Ry(theta_i) on qubit i
            CNOT chain for entanglement
        Measure all qubits in the Z basis.
    The expectation value of the product of all Z measurements is returned.
    """

    def __init__(self, n_qubits: int, depth: int = 1, shots: int = 1024) -> None:
        self.n_qubits = n_qubits
        self.depth = depth
        self.shots = shots
        self.backend = Aer.get_backend("qasm_simulator")
        self._build_circuit()

    def _build_circuit(self) -> None:
        """Create a parameterised circuit with a ParameterVector."""
        self.qreg = QuantumRegister(self.n_qubits, "q")
        self.creg = ClassicalRegister(self.n_qubits, "c")
        self.circuit = QuantumCircuit(self.qreg, self.creg)

        # Total number of parameters: n_qubits * depth
        self.params = ParameterVector("theta", self.n_qubits * self.depth)

        # Build layers
        param_idx = 0
        for _ in range(self.depth):
            for q in range(self.n_qubits):
                self.circuit.ry(self.params[param_idx], q)
                param_idx += 1
            # Entangling CNOT chain (linear)
            for q in range(self.n_qubits - 1):
                self.circuit.cx(q, q + 1)

        self.circuit.measure(self.qreg, self.creg)

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Execute the circuit for a batch of parameter sets.

        Parameters
        ----------
        thetas : Iterable[float]
            Iterable of parameter values. Length must equal n_qubits * depth.
            Each batch element is interpreted as a single parameter set.

        Returns
        -------
        np.ndarray
            1‑D array of expectation values, one per parameter set.
        """
        # Ensure proper shape
        theta_list = np.asarray(list(thetas), dtype=np.float64)
        if theta_list.ndim == 1:
            theta_list = theta_list.reshape(-1, self.n_qubits * self.depth)

        expectations = []
        for theta_set in theta_list:
            bind_dict = {param: val for param, val in zip(self.params, theta_set)}
            bound_circ = self.circuit.bind_parameters(bind_dict)

            job = execute(bound_circ, self.backend, shots=self.shots)
            result = job.result()
            counts = result.get_counts(bound_circ)

            # Compute expectation of product of Z (parity)
            exp_val = 0.0
            for bitstring, cnt in counts.items():
                # bitstring is '01...' with most significant bit first
                parity = (-1) ** (bitstring.count('1'))
                exp_val += parity * cnt / self.shots
            expectations.append(exp_val)

        return np.array(expectations)

    def parameter_shift_gradient(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Estimate gradient of the expectation w.r.t. each parameter using the
        parameter‑shift rule.

        Parameters
        ----------
        thetas : Iterable[float]
            Parameter set for which to compute the gradient.

        Returns
        -------
        np.ndarray
            Gradient vector of shape (n_qubits*depth,).
        """
        theta_array = np.asarray(list(thetas), dtype=np.float64)
        grad = np.zeros_like(theta_array)

        shift = np.pi / 2
        for i, param in enumerate(self.params):
            # +shift
            theta_plus = theta_array.copy()
            theta_plus[i] += shift
            exp_plus = self.run(theta_plus)[0]

            # -shift
            theta_minus = theta_array.copy()
            theta_minus[i] -= shift
            exp_minus = self.run(theta_minus)[0]

            grad[i] = 0.5 * (exp_plus - exp_minus)

        return grad


__all__ = ["QuantumCircuitLayer"]
