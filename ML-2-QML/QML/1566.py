"""Quantum self‑attention with amplitude encoding and parameterised rotations."""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.providers.aer import AerSimulator
from qiskit.circuit import ParameterVector


class SelfAttentionGen128:
    """
    Quantum self‑attention circuit that encodes classical inputs into qubit
    states, applies rotation layers driven by ``rotation_params``, entangles
    adjacent qubits via controlled‑rotation gates driven by ``entangle_params``,
    and measures all qubits.  The design supports hybrid training via
    gradient‑based optimisers by exposing a clear parameter interface.
    """

    def __init__(self, n_qubits: int, backend: qiskit.providers.Backend | None = None):
        """
        Parameters
        ----------
        n_qubits : int
            Number of qubits used for the attention block.
        backend : qiskit.providers.Backend, optional
            Execution backend.  Defaults to AerSimulator('qasm_simulator').
        """
        self.n_qubits = n_qubits
        self.backend = backend or AerSimulator()
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")

    def _build_circuit(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
    ) -> QuantumCircuit:
        """
        Assemble the full circuit for a single attention pass.

        Parameters
        ----------
        rotation_params : np.ndarray
            Length 3 * n_qubits array of rotation angles for RX, RY, RZ.
        entangle_params : np.ndarray
            Length (n_qubits - 1) array of controlled‑rotation angles.
        inputs : np.ndarray
            Classical input vector of length n_qubits used for amplitude
            encoding via Ry gates.

        Returns
        -------
        QuantumCircuit
            The assembled circuit ready for execution.
        """
        circuit = QuantumCircuit(self.qr, self.cr)

        # Amplitude encoding: map input values to rotation angles
        for i, val in enumerate(inputs):
            circuit.ry(val, i)

        # Parameterised rotation layer
        rot_vec = ParameterVector("rot", 3 * self.n_qubits)
        for i in range(self.n_qubits):
            idx = 3 * i
            circuit.rx(rot_vec[idx], i)
            circuit.ry(rot_vec[idx + 1], i)
            circuit.rz(rot_vec[idx + 2], i)

        # Entanglement via controlled‑rotation gates
        ent_vec = ParameterVector("ent", self.n_qubits - 1)
        for i in range(self.n_qubits - 1):
            circuit.crx(ent_vec[i], i, i + 1)

        # Bind the provided parameters
        param_bindings = {
            **{rot_vec[j]: rotation_params[j] for j in range(3 * self.n_qubits)},
            **{ent_vec[j]: entangle_params[j] for j in range(self.n_qubits - 1)},
        }
        circuit = circuit.bind_parameters(param_bindings)

        # Measurement
        circuit.measure(self.qr, self.cr)
        return circuit

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
        shots: int = 1024,
    ) -> dict[str, int]:
        """
        Execute the attention circuit on the configured backend.

        Parameters
        ----------
        rotation_params : np.ndarray
            Rotation angles for the parameterised layer.
        entangle_params : np.ndarray
            Entanglement angles for controlled‑rotation gates.
        inputs : np.ndarray
            Classical input vector for amplitude encoding.
        shots : int, optional
            Number of shots for the simulation.  Defaults to 1024.

        Returns
        -------
        dict[str, int]
            Measurement outcome counts.
        """
        circuit = self._build_circuit(rotation_params, entangle_params, inputs)
        job = qiskit.execute(circuit, self.backend, shots=shots)
        return job.result().get_counts(circuit)


__all__ = ["SelfAttentionGen128"]
