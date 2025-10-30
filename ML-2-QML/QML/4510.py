"""Quantum hybrid estimator that mirrors the classical HybridEstimatorQNN.

The circuit consists of:
* an auto‑encoder ansatz built with RealAmplitudes,
* a domain‑wall swap‑test for latent reconstruction,
* a fixed attention‑style entangling block,
* measurement of a single auxiliary qubit.
The estimator uses Qiskit’s StatevectorEstimator to obtain expectation values.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute
from qiskit.circuit.library import RealAmplitudes
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import StatevectorEstimator


class HybridEstimatorQNN:
    """
    Quantum implementation of the hybrid estimator.

    The class builds a variational circuit that embeds the same logical flow as the
    classical model: an auto‑encoder ansatz, a swap‑test for latent reconstruction,
    and a simple attention‑style entanglement pattern.  The estimator returns
    expectation values of a single Pauli‑Z observable on the auxiliary qubit,
    which can be interpreted as a regression score.
    """

    def __init__(
        self,
        num_latent: int = 3,
        num_trash: int = 2,
        attention_depth: int = 1,
    ) -> None:
        self.num_latent = num_latent
        self.num_trash = num_trash
        self.attention_depth = attention_depth
        self.circuit = self._build_circuit()
        self.estimator = StatevectorEstimator()
        self.qnn = EstimatorQNN(
            circuit=self.circuit,
            observables=self._build_observable(),
            input_params=[],
            weight_params=self.circuit.parameters,
            estimator=self.estimator,
        )

    def _build_circuit(self) -> QuantumCircuit:
        """Construct the full variational circuit."""
        total_qubits = self.num_latent + 2 * self.num_trash + 1
        qr = QuantumRegister(total_qubits, "q")
        cr = ClassicalRegister(1, "c")
        circuit = QuantumCircuit(qr, cr)

        # Auto‑encoder ansatz
        ansatz = RealAmplitudes(self.num_latent + self.num_trash, reps=5)
        circuit.compose(ansatz, range(0, self.num_latent + self.num_trash), inplace=True)
        circuit.barrier()

        # Swap‑test for latent reconstruction
        aux = self.num_latent + 2 * self.num_trash
        circuit.h(aux)
        for i in range(self.num_trash):
            circuit.cswap(aux, self.num_latent + i, self.num_latent + self.num_trash + i)
        circuit.h(aux)

        # Attention‑style entangling block
        for _ in range(self.attention_depth):
            for i in range(total_qubits - 1):
                circuit.crx(np.pi / 4, i, i + 1)

        circuit.measure(aux, cr[0])
        return circuit

    def _build_observable(self) -> SparsePauliOp:
        """Pauli‑Z on the auxiliary qubit."""
        return SparsePauliOp.from_list([("Z" * self.circuit.num_qubits, 1)])

    def run(self, backend=None, shots: int = 1024) -> dict:
        """
        Execute the circuit and return measurement counts.

        Parameters
        ----------
        backend : qiskit.providers.Backend, optional
            If None, the Aer qasm simulator is used.
        shots : int
            Number of shots for the execution.
        """
        if backend is None:
            backend = Aer.get_backend("qasm_simulator")
        job = execute(self.circuit, backend, shots=shots)
        return job.result().get_counts(self.circuit)
