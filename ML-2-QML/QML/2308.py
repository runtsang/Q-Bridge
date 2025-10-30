"""Hybrid quantum layer combining a parameterized Ry circuit with a quanvolution filter.

The quantum implementation builds two distinct sub‑circuits:
1. A fully‑connected layer circuit that applies a Ry gate to each qubit,
   parameterised by ``theta_fc``.
2. A quanvolution circuit that maps a 2‑D kernel onto a set of qubits,
   parameterised by the input data and a random circuit.

The ``run`` method accepts a flattened list of parameters for the FC circuit
followed by the data for the quanvolution circuit.  It returns a tuple
containing the expectation value of the FC circuit and the average
probability of measuring |1> across the quanvolution qubits.
"""

import numpy as np
import qiskit
from qiskit.circuit.random import random_circuit


def FCLConvQuantum(
    n_qubits_fc: int = 1,
    kernel_size: int = 2,
    backend=None,
    shots: int = 100,
    threshold: float = 127,
) -> qiskit.QuantumCircuit:
    if backend is None:
        backend = qiskit.Aer.get_backend("qasm_simulator")

    # Fully‑connected circuit
    fc_circ = qiskit.QuantumCircuit(n_qubits_fc)
    theta_fc = [qiskit.circuit.Parameter(f"theta_fc_{i}") for i in range(n_qubits_fc)]
    for i, theta in enumerate(theta_fc):
        fc_circ.ry(theta, i)
    fc_circ.barrier()
    fc_circ.measure_all()

    # Quanvolution circuit
    n_qubits_qv = kernel_size ** 2
    qv_circ = qiskit.QuantumCircuit(n_qubits_qv)
    theta_qv = [qiskit.circuit.Parameter(f"theta_qv_{i}") for i in range(n_qubits_qv)]
    for i, theta in enumerate(theta_qv):
        qv_circ.rx(theta, i)
    qv_circ.barrier()
    qv_circ += random_circuit(n_qubits_qv, 2)
    qv_circ.measure_all()

    class HybridQuantumLayer:
        def __init__(self) -> None:
            self.backend = backend
            self.shots = shots
            self.threshold = threshold

        def run(self, thetas: Iterable[float], data: np.ndarray) -> tuple[float, float]:
            # Bind FC parameters
            fc_binds = [{theta: val for theta, val in zip(theta_fc, thetas[:n_qubits_fc])}]

            # Bind QV parameters based on data
            data_flat = np.reshape(data, (1, n_qubits_qv))
            qv_binds = []
            for dat in data_flat:
                bind = {}
                for i, val in enumerate(dat):
                    bind[theta_qv[i]] = np.pi if val > self.threshold else 0
                qv_binds.append(bind)

            # Execute FC circuit
            job_fc = qiskit.execute(
                fc_circ,
                self.backend,
                shots=self.shots,
                parameter_binds=fc_binds,
            )
            result_fc = job_fc.result().get_counts(fc_circ)
            probs_fc = np.array(list(result_fc.values())) / self.shots
            expectation_fc = np.sum(np.array(list(result_fc.keys()), dtype=float) * probs_fc)

            # Execute QV circuit
            job_qv = qiskit.execute(
                qv_circ,
                self.backend,
                shots=self.shots,
                parameter_binds=qv_binds,
            )
            result_qv = job_qv.result().get_counts(qv_circ)

            # Compute average probability of measuring |1> across all qubits
            total_ones = 0
            for key, val in result_qv.items():
                total_ones += sum(int(bit) for bit in key) * val
            prob_ones = total_ones / (self.shots * n_qubits_qv)

            return expectation_fc, prob_ones

    return HybridQuantumLayer()


__all__ = ["FCLConvQuantum"]
