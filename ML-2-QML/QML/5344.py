"""Quantum‑classical hybrid network that mirrors HybridConvNet.

The :class:`HybridConvNet` below builds a composite quantum circuit
that first applies a 2‑qubit quantum convolution, then a small
auto‑encoder circuit, and finally a sampler or estimator QNN.
The class exposes a :meth:`run` method that accepts a 2×2 input array
and returns the measurement statistics of the last circuit layer.
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit.random import random_circuit
from qiskit.circuit.library import RealAmplitudes
from qiskit_machine_learning.neural_networks import SamplerQNN, EstimatorQNN
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit.primitives import StatevectorEstimator as Estimator
from qiskit.quantum_info import SparsePauliOp

# ---------- quantum convolution ----------
class QuanvCircuit:
    """2‑qubit quantum convolution filter."""

    def __init__(self, kernel_size: int, backend, shots: int, threshold: float) -> None:
        self.n_qubits = kernel_size ** 2
        self._circuit = QuantumCircuit(self.n_qubits)
        self.theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            self._circuit.rx(self.theta[i], i)
        self._circuit.barrier()
        self._circuit += random_circuit(self.n_qubits, 2)
        self._circuit.measure_all()
        self.backend = backend
        self.shots = shots
        self.threshold = threshold

    def run(self, data: np.ndarray) -> float:
        """Execute the quantum conv on a 2×2 data array."""
        data = np.reshape(data, (1, self.n_qubits))
        param_binds = []
        for dat in data:
            bind = {}
            for i, val in enumerate(dat):
                bind[self.theta[i]] = np.pi if val > self.threshold else 0
            param_binds.append(bind)

        job = execute(
            self._circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=param_binds,
        )
        result = job.result()
        counts = result.get_counts(self._circuit)
        # average number of |1> outcomes
        total = 0
        for key, val in counts.items():
            ones = sum(int(bit) for bit in key)
            total += ones * val
        return total / (self.shots * self.n_qubits)

# ---------- auto‑encoder circuit ----------
def auto_encoder_circuit(num_latent: int, num_trash: int) -> QuantumCircuit:
    """Return a simple auto‑encoder circuit used in the hybrid model."""
    qr = qiskit.QuantumRegister(num_latent + 2 * num_trash + 1, "q")
    cr = qiskit.ClassicalRegister(1, "c")
    circuit = QuantumCircuit(qr, cr)
    circuit.compose(RealAmplitudes(num_latent + num_trash, reps=5), range(0, num_latent + num_trash), inplace=True)
    circuit.barrier()
    aux = num_latent + 2 * num_trash
    circuit.h(aux)
    for i in range(num_trash):
        circuit.cswap(aux, num_latent + i, num_latent + num_trash + i)
    circuit.h(aux)
    circuit.measure(aux, cr[0])
    return circuit

# ---------- hybrid network ----------
class HybridConvNet:
    """Quantum‑classical hybrid network that composes a quantum conv,
    an auto‑encoder, and a sampler or estimator QNN.
    """

    def __init__(
        self,
        backend=None,
        shots: int = 100,
        conv_threshold: float = 127,
        classifier: str = "sampler",
    ):
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.shots = shots
        self.conv = QuanvCircuit(kernel_size=2, backend=self.backend, shots=shots, threshold=conv_threshold)
        self.autoencoder = auto_encoder_circuit(num_latent=3, num_trash=2)

        if classifier == "sampler":
            # build a simple SamplerQNN using a 2‑qubit circuit
            inputs = qiskit.circuit.ParameterVector("input", 2)
            weights = qiskit.circuit.ParameterVector("weight", 4)
            qc = QuantumCircuit(2)
            qc.ry(inputs[0], 0)
            qc.ry(inputs[1], 1)
            qc.cx(0, 1)
            qc.ry(weights[0], 0)
            qc.ry(weights[1], 1)
            qc.cx(0, 1)
            qc.ry(weights[2], 0)
            qc.ry(weights[3], 1)
            sampler = Sampler()
            self.qnn = SamplerQNN(circuit=qc, input_params=inputs, weight_params=weights, sampler=sampler)
        else:
            # build a simple EstimatorQNN using a 1‑qubit circuit
            inp = qiskit.circuit.Parameter("input1")
            wgt = qiskit.circuit.Parameter("weight1")
            qc = QuantumCircuit(1)
            qc.h(0)
            qc.ry(inp, 0)
            qc.rx(wgt, 0)
            observable = SparsePauliOp.from_list([("Y", 1)])
            estimator = Estimator()
            self.qnn = EstimatorQNN(circuit=qc, observables=observable, input_params=[inp], weight_params=[wgt], estimator=estimator)

    def run(self, data: np.ndarray) -> dict:
        """Run the full hybrid circuit on a 2×2 input array.

        Parameters
        ----------
        data
            2×2 array of classical pixel values.

        Returns
        -------
        dict
            Raw measurement results from the last layer of the network.
        """
        # 1. quantum convolution
        conv_out = self.conv.run(data)

        # 2. auto‑encoder (no parameters, just a fixed circuit)
        ae_circ = self.autoencoder

        # 3. compose all circuits
        full_circuit = self.conv._circuit.compose(ae_circ, inplace=False)
        full_circuit = full_circuit.compose(self.qnn.circuit, inplace=False)

        # 4. execute
        job = execute(full_circuit, self.backend, shots=self.shots)
        return job.result().get_counts(full_circuit)

__all__ = ["HybridConvNet"]
