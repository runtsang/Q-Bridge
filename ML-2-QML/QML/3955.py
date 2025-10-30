"""Quantum version of HybridAutoEncoder.

The implementation builds on the swap‑test + domain‑wall auto‑encoder from the
original seed and adds a tiny EstimatorQNN‑style sub‑network for regression.
Everything is expressed with Qiskit primitives, making it suitable for
simulation or execution on a real backend.

Key features:
* ``encode`` – runs the auto‑encoder circuit and returns the probability of
  measuring 1 on the ancillary qubit.
* ``estimate`` – uses a 1‑qubit EstimatorQNN circuit to produce a scalar output.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Parameter
from qiskit.circuit.library import RealAmplitudes
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN, SamplerQNN
from qiskit.primitives import StatevectorSampler, StatevectorEstimator


class HybridAutoEncoder:
    """Quantum auto‑encoder + quantum estimator."""

    def __init__(
        self,
        latent_dim: int = 3,
        num_trash: int = 2,
        reps: int = 5,
        backend: str = "statevector_simulator",
    ) -> None:
        self.latent_dim = latent_dim
        self.num_trash = num_trash
        self.reps = reps
        self.backend = backend

        # Sampler for the auto‑encoder
        self.sampler = StatevectorSampler(backend=self.backend)
        # Estimator for the regression sub‑network
        self.estimator = StatevectorEstimator(backend=self.backend)

        # Pre‑build the auto‑encoder circuit
        self.auto_circuit = self._build_auto_circuit()

    # --------------------------------------------------------------------- #
    # Auto‑encoder circuit construction
    # --------------------------------------------------------------------- #
    def _ansatz(self, num_qubits: int) -> QuantumCircuit:
        return RealAmplitudes(num_qubits, reps=self.reps)

    def _build_auto_circuit(self) -> QuantumCircuit:
        qr = QuantumRegister(self.latent_dim + 2 * self.num_trash + 1, "q")
        cr = ClassicalRegister(1, "c")
        qc = QuantumCircuit(qr, cr)

        # Encode latent + trash with the ansatz
        qc.compose(
            self._ansatz(self.latent_dim + self.num_trash),
            range(0, self.latent_dim + self.num_trash),
            inplace=True,
        )
        qc.barrier()

        # Domain‑wall + swap‑test
        aux = self.latent_dim + 2 * self.num_trash
        qc.h(aux)
        for i in range(self.num_trash):
            qc.cswap(aux, self.latent_dim + i, self.latent_dim + self.num_trash + i)
        qc.h(aux)
        qc.measure(aux, cr[0])

        return qc

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    def encode(self, inputs: np.ndarray) -> np.ndarray:
        """Run the auto‑encoder and return the probability of measuring 1."""
        # Simple embedding: rotate the first qubit by each input value
        qc = self.auto_circuit.copy()
        for idx, val in enumerate(inputs):
            qc.ry(val, idx % (self.latent_dim + self.num_trash))
        result = self.sampler.run(qc, shots=1024).result()
        counts = result.get_counts()
        prob_one = counts.get("1", 0) / 1024
        return np.array([prob_one])

    # --------------------------------------------------------------------- #
    # Estimator sub‑network
    # --------------------------------------------------------------------- #
    def _estimator_qnn(self) -> EstimatorQNN:
        """Builds a 1‑qubit EstimatorQNN with a Y observable."""
        input_param = Parameter("input")
        weight_param = Parameter("weight")
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.ry(input_param, 0)
        qc.rx(weight_param, 0)

        observable = SparsePauliOp.from_list([("Y", 1)])
        return EstimatorQNN(
            circuit=qc,
            observables=observable,
            input_params=[input_param],
            weight_params=[weight_param],
            estimator=self.estimator,
        )

    def estimate(self, inputs: np.ndarray) -> float:
        """Return a scalar prediction using the quantum EstimatorQNN."""
        qnn = self._estimator_qnn()
        # Map the first input value to the rotation angle
        param_values = {qnn.input_params[0]: inputs[0]}
        # Optimize the weight for a single prediction (COBYLA with a single step)
        opt = COBYLA()
        def objective(params):
            param_values[qnn.weight_params[0]] = params[0]
            return -qnn(param_values).item()  # maximize expectation

        init_guess = np.array([0.0])
        opt_result = opt.minimize(objective, init_guess, options={"maxiter": 1})
        best_weight = opt_result.x[0]
        param_values[qnn.weight_params[0]] = best_weight
        return qnn(param_values).item()

__all__ = ["HybridAutoEncoder"]
