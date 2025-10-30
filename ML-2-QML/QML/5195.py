"""Hybrid quantum network that mirrors the classical `HybridFCL` architecture.

The implementation stitches together a parameterised convolution circuit,
a quantum auto‑encoder, a simple fully‑connected variational circuit, and
an EstimatorQNN.  Each sub‑circuit is executed on the Aer simulator
and the resulting expectation values are fed into the EstimatorQNN,
which performs the final regression step.  The module exposes a
`HybridFCL` class with a `run` method that accepts a 1‑D array of
classical data and returns a single‑dimensional prediction.
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute
from qiskit.circuit import Parameter
from qiskit.circuit.library import RealAmplitudes
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.random import random_circuit

# --------------------------------------------------------------------------- #
# Helper functions for building sub‑circuits
# --------------------------------------------------------------------------- #
def _conv_circuit(kernel_size: int = 2, threshold: float = 0.5) -> QuantumCircuit:
    n_qubits = kernel_size ** 2
    qc = QuantumCircuit(n_qubits)
    theta = [Parameter(f"theta{i}") for i in range(n_qubits)]
    for i in range(n_qubits):
        qc.rx(theta[i], i)
    qc.barrier()
    qc += random_circuit(n_qubits, 2)
    qc.measure_all()
    return qc

def _autoencoder_circuit(num_latent: int = 3, num_trash: int = 2) -> QuantumCircuit:
    qr = QuantumRegister(num_latent + 2 * num_trash + 1, "q")
    cr = ClassicalRegister(1, "c")
    qc = QuantumCircuit(qr, cr)
    # Ansatz
    ansatz = RealAmplitudes(num_latent + num_trash, reps=5)
    qc.compose(ansatz, range(num_latent + num_trash), inplace=True)
    qc.barrier()
    aux = num_latent + 2 * num_trash
    qc.h(aux)
    for i in range(num_trash):
        qc.cswap(aux, num_latent + i, num_latent + num_trash + i)
    qc.h(aux)
    qc.measure(aux, cr[0])
    return qc

def _fcl_circuit() -> QuantumCircuit:
    qc = QuantumCircuit(1)
    theta = Parameter("theta")
    qc.h(0)
    qc.ry(theta, 0)
    qc.measure_all()
    return qc

def _estimator_qnn() -> EstimatorQNN:
    params = [Parameter("input1"), Parameter("weight1")]
    qc = QuantumCircuit(1)
    qc.h(0)
    qc.ry(params[0], 0)
    qc.rx(params[1], 0)
    observable = SparsePauliOp.from_list([("Y", 1)])
    estimator = StatevectorEstimator()
    return EstimatorQNN(circuit=qc,
                        observables=observable,
                        input_params=[params[0]],
                        weight_params=[params[1]],
                        estimator=estimator)

# --------------------------------------------------------------------------- #
# Hybrid quantum network
# --------------------------------------------------------------------------- #
class HybridFCL:
    """
    Quantum counterpart of the classical `HybridFCL`.  The class builds four
    sub‑circuits and executes them sequentially.  The outputs of the first
    three circuits are used as the two inputs to the EstimatorQNN, which
    produces the final regression value.
    """
    def __init__(self, seed: int = 42) -> None:
        np.random.seed(seed)
        self.backend = Aer.get_backend("qasm_simulator")
        self.shots = 100
        self.conv = _conv_circuit()
        self.autoencoder = _autoencoder_circuit()
        self.fcl = _fcl_circuit()
        self.estimator = _estimator_qnn()

    def _run_circuit(self, qc: QuantumCircuit, data: np.ndarray | list | tuple,
                     params: list[Parameter] | None = None) -> float:
        """Execute a circuit with optional parameter binding and return
        the mean number of |1⟩ outcomes per qubit."""
        data = np.asarray(data).reshape((1, -1))[0]
        if params is not None:
            bind = {p: np.pi if val > 0.5 else 0.0 for p, val in zip(params, data)}
            job = execute(qc, self.backend, shots=self.shots,
                          parameter_binds=[bind])
        else:
            job = execute(qc, self.backend, shots=self.shots)
        result = job.result().get_counts(qc)
        total = 0
        for bits, cnt in result.items():
            ones = sum(int(b) for b in bits)
            total += ones * cnt
        return total / (self.shots * qc.num_qubits)

    def run(self, data: np.ndarray | list | tuple) -> np.ndarray:
        """Run the full quantum pipeline and return the EstimatorQNN output."""
        # Convolution output
        conv_val = self._run_circuit(self.conv, data, self.conv.parameters)
        # Auto‑encoder output
        ae_val = self._run_circuit(self.autoencoder, data, self.autoencoder.parameters)
        # Fully‑connected variational output (use a fixed theta for simplicity)
        fcl_val = self._run_circuit(self.fcl, [0.0], [self.fcl.parameters[0]])
        # EstimatorQNN prediction
        preds = self.estimator.predict(np.array([[conv_val]]),
                                      weight_params=np.array([[fcl_val]]))
        return preds

def FCL() -> HybridFCL:
    """Factory that mirrors the classical helper."""
    return HybridFCL()

__all__ = ["FCL", "HybridFCL"]
