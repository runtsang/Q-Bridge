from __future__ import annotations

import numpy as np
from qiskit import Aer, QuantumCircuit
from qiskit.circuit import Parameter
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import Estimator as StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp


class HybridQuantumEstimatorQNN:
    """
    Composite quantum neural network that concatenates the EstimatorQNN circuit
    with a fully‑connected parameterised quantum circuit (FCL). The combined
    circuit is passed to a Qiskit EstimatorQNN, enabling end‑to‑end training
    with a quantum back‑propagation algorithm.
    """

    def __init__(self, n_qubits: int = 1, shots: int = 1024) -> None:
        self.n_qubits = n_qubits
        self.shots = shots
        self.backend = Aer.get_backend("qasm_simulator")
        self._build_circuits()

    def _build_circuits(self) -> None:
        # EstimatorQNN sub‑circuit
        self.input_param = Parameter("input")
        self.weight_param = Parameter("weight")
        self.qc_estimator = QuantumCircuit(1)
        self.qc_estimator.h(0)
        self.qc_estimator.ry(self.input_param, 0)
        self.qc_estimator.rx(self.weight_param, 0)

        # FCL sub‑circuit
        self.fcl_param = Parameter("theta")
        self.qc_fcl = QuantumCircuit(1)
        self.qc_fcl.h(0)
        self.qc_fcl.barrier()
        self.qc_fcl.ry(self.fcl_param, 0)
        self.qc_fcl.measure_all()

        # Combine the two sub‑circuits into a single circuit
        self.combined_circuit = QuantumCircuit(1)
        self.combined_circuit.append(self.qc_estimator, [0])
        self.combined_circuit.append(self.qc_fcl, [0])

    def run(self, thetas: np.ndarray) -> np.ndarray:
        """
        Evaluate the combined circuit with a single set of parameters.

        Parameters
        ----------
        thetas : np.ndarray
            Array of shape (3,) containing [input, weight, theta].
        """
        param_binds = [
            {
                self.input_param: thetas[0],
                self.weight_param: thetas[1],
                self.fcl_param: thetas[2],
            }
        ]
        job = self.backend.run(
            self.combined_circuit,
            shots=self.shots,
            parameter_binds=param_binds,
        )
        result = job.result()
        counts = result.get_counts(self.combined_circuit)
        probs = np.array(list(counts.values())) / self.shots
        states = np.array([int(k, 2) for k in counts.keys()], dtype=float)
        expectation = np.sum(states * probs)
        return np.array([expectation])

    def estimator(self) -> EstimatorQNN:
        """
        Return a Qiskit EstimatorQNN that wraps the combined circuit.
        """
        observable = SparsePauliOp.from_list([("Y" * self.n_qubits, 1)])
        estimator = StatevectorEstimator()
        return EstimatorQNN(
            circuit=self.combined_circuit,
            observables=observable,
            input_params=[self.input_param],
            weight_params=[self.weight_param, self.fcl_param],
            estimator=estimator,
        )


__all__ = ["HybridQuantumEstimatorQNN"]
