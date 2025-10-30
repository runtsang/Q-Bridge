"""Quantum hybrid model mirroring HybridFCL.

The circuit follows the same logical stages as the classical counterpart:

1. Feature encoding via RX gates.
2. Attention‑style entanglement with CRX gates.
3. Variational layers consisting of Ry rotations and CZ entanglement.
4. Measurement of single‑qubit Z observables to produce logits.

The :class:`HybridFCL` class exposes a ``run`` method that accepts a NumPy
array of inputs and returns logits.  All parameters are bound per sample
and the circuit is executed on Qiskit's Aer simulator.
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector, Parameter
from qiskit.quantum_info import SparsePauliOp

__all__ = ["HybridFCL"]


class HybridFCL:
    """
    Quantum hybrid model that emulates the classical HybridFCL.

    Parameters
    ----------
    n_features : int
        Number of input features (also the number of qubits).
    depth : int, default 2
        Depth of the variational ansatz.
    n_classes : int, default 2
        Number of output classes.
    shots : int, default 1024
        Number of shots for simulation.
    """

    def __init__(
        self,
        n_features: int,
        depth: int = 2,
        n_classes: int = 2,
        shots: int = 1024,
    ) -> None:
        self.n_features = n_features
        self.depth = depth
        self.n_classes = n_classes
        self.shots = shots
        self.backend = qiskit.Aer.get_backend("qasm_simulator")

        # Build circuit once; parameters are bound for each sample
        self.circuit, self.encoding, self.weights, self.attn_params, self.observables = (
            self._build_circuit()
        )

    def _build_circuit(self):
        # Parameter vectors
        enc = ParameterVector("x", self.n_features)
        wts = ParameterVector("theta", self.n_features * self.depth)
        attn = ParameterVector("phi", self.n_features - 1)

        circ = QuantumCircuit(self.n_features)

        # 1. Feature encoding
        for qubit, param in enumerate(enc):
            circ.rx(param, qubit)

        # 2. Attention‑style entanglement
        for i, param in enumerate(attn):
            circ.crx(param, i, i + 1)

        # 3. Variational layers
        w_idx = 0
        for _ in range(self.depth):
            for qubit in range(self.n_features):
                circ.ry(wts[w_idx], qubit)
                w_idx += 1
            for qubit in range(self.n_features - 1):
                circ.cz(qubit, qubit + 1)

        # 4. Observables for classification
        obs = [
            SparsePauliOp("I" * i + "Z" + "I" * (self.n_features - i - 1))
            for i in range(self.n_features)
        ]

        return circ, enc, wts, attn, obs

    def run(self, inputs: np.ndarray) -> np.ndarray:
        """
        Execute the circuit for a batch of inputs.

        Parameters
        ----------
        inputs : np.ndarray
            Shape (batch, n_features).

        Returns
        -------
        np.ndarray
            Logits of shape (batch, n_classes).
        """
        batch_logits = []

        for sample in inputs:
            # Bind encoding parameters to the sample
            param_bindings: dict[Parameter, float] = {
                p: v for p, v in zip(self.encoding, sample)
            }

            # Initialise variational and attention weights to zero
            for p in self.weights:
                param_bindings[p] = 0.0
            for p in self.attn_params:
                param_bindings[p] = 0.0

            bound_circ = self.circuit.bind_parameters(param_bindings)

            job = qiskit.execute(bound_circ, self.backend, shots=self.shots)
            result = job.result()

            # Expectation values of the observables
            exp_vals = []
            for obs in self.observables:
                exp = result.get_expectation_value(obs, bound_circ)
                exp_vals.append(exp)
            exp_vals = np.array(exp_vals)

            # Reduce to the requested number of classes
            if self.n_classes == 2:
                logits = np.array([exp_vals.sum(), -exp_vals.sum()])
            else:
                logits = exp_vals[: self.n_classes]
            batch_logits.append(logits)

        return np.vstack(batch_logits)
