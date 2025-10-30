"""Variational quantum circuit implementation of a fully connected layer.

The circuit uses a stack of parameterized single‑qubit rotations
followed by entangling CNOTs.  A single Pauli‑Z expectation
value is returned for each input vector of parameters.
"""

import numpy as np
import qiskit
from qiskit import QuantumCircuit, execute, Aer
from qiskit.circuit import Parameter
from qiskit.providers import Backend
from typing import Iterable, List


class FCL:
    """
    Variational quantum circuit mimicking a fully‑connected layer.

    Parameters
    ----------
    n_qubits : int
        Number of qubits, equal to the number of input parameters.
    layers : int, optional
        Number of variational layers (each contains a rotation per qubit
        and an entangling layer).
    backend : Backend or str, optional
        Qiskit backend to execute on.  If a string, a simulator with that
        name is fetched.
    shots : int, optional
        Number of shots per execution.
    """

    def __init__(
        self,
        n_qubits: int,
        layers: int = 2,
        backend: Backend | str = "qasm_simulator",
        shots: int = 1024,
    ) -> None:
        self.n_qubits = n_qubits
        self.layers = layers
        self.shots = shots

        if isinstance(backend, str):
            self.backend = Aer.get_backend(backend)
        else:
            self.backend = backend

        # Build symbolic circuit once
        self.circuit = QuantumCircuit(n_qubits)
        self.params: List[Parameter] = []

        for lay in range(layers):
            # Parameterized Ry rotations on each qubit
            for q in range(n_qubits):
                theta = Parameter(f"θ_{lay}_{q}")
                self.params.append(theta)
                self.circuit.ry(theta, q)

            # Entangling layer: CNOT chain
            for q in range(n_qubits - 1):
                self.circuit.cx(q, q + 1)

        # Measurement in Z basis
        self.circuit.barrier()
        self.circuit.measure_all()

    def _bind_params(self, thetas: Iterable[float]) -> List[dict]:
        """
        Create a list of parameter bindings for a batch of input vectors.

        Parameters
        ----------
        thetas : Iterable[float]
            Flattened parameter vector of length n_qubits * layers.

        Returns
        -------
        List[dict]
            List of dictionaries mapping Parameter -> float.
        """
        theta_list = list(thetas)
        if len(theta_list)!= len(self.params):
            raise ValueError(
                f"Expected {len(self.params)} parameters, got {len(theta_list)}."
            )
        return [{p: v for p, v in zip(self.params, theta_list)}]

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Execute the circuit for a single input vector and return the
        expectation value of the Pauli‑Z measurement.

        Parameters
        ----------
        thetas : Iterable[float]
            Flattened vector of rotation angles.

        Returns
        -------
        np.ndarray
            Array of shape (1,) containing the expectation value.
        """
        bind_dicts = self._bind_params(thetas)

        job = execute(
            self.circuit,
            backend=self.backend,
            shots=self.shots,
            parameter_binds=bind_dicts,
        )
        result = job.result()
        counts = result.get_counts(self.circuit)

        # Convert counts to expectation of Z (|0> -> +1, |1> -> -1)
        exp_val = 0.0
        total = sum(counts.values())
        for state, cnt in counts.items():
            # state string is bitstring; interpret as integer
            val = 1.0 if int(state, 2) % 2 == 0 else -1.0  # single qubit expectation
            exp_val += val * cnt
        exp_val /= total

        return np.array([exp_val])

    def parameter_shift_gradient(
        self,
        thetas: Iterable[float],
        shift: float = np.pi / 2,
    ) -> np.ndarray:
        """
        Estimate the gradient of the expectation value w.r.t. all parameters
        using the parameter‑shift rule.

        Parameters
        ----------
        thetas : Iterable[float]
            Current parameter vector.
        shift : float, optional
            Shift amount.  Default is π/2.

        Returns
        -------
        np.ndarray
            Gradient array of shape (n_params,).
        """
        theta_vec = np.array(thetas, dtype=float)
        grads = np.zeros_like(theta_vec)

        for idx in range(len(theta_vec)):
            perturbed_plus = theta_vec.copy()
            perturbed_minus = theta_vec.copy()
            perturbed_plus[idx] += shift
            perturbed_minus[idx] -= shift

            f_plus = self.run(perturbed_plus)[0]
            f_minus = self.run(perturbed_minus)[0]

            grads[idx] = 0.5 * (f_plus - f_minus)

        return grads
