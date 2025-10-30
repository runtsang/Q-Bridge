"""Hybrid quantum autoencoder with self‑attention and a quantum estimator head.

The implementation builds upon the original quantum autoencoder circuit, the
quantum self‑attention block, and the EstimatorQNN example.  The class exposes
the same factory signature, allowing existing code that expects an
`Autoencoder`‑style object to instantiate a purely quantum model with
minimal changes.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, Aer, QuantumRegister, ClassicalRegister
from qiskit.circuit import Parameter
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Sampler, Estimator
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import SamplerQNN, EstimatorQNN
from qiskit.algorithms.optimizers import COBYLA


# --------------------------- Quantum Autoencoder ---------------------------

def _autoencoder_circuit(num_latent: int, num_trash: int) -> QuantumCircuit:
    """Construct swap‑test autoencoder circuit used in the sampler QNN."""
    qr = QuantumRegister(num_latent + 2 * num_trash + 1, "q")
    cr = ClassicalRegister(1, "c")
    circuit = QuantumCircuit(qr, cr)

    # Encode: variational ansatz on latent + trash qubits
    circuit.compose(RealAmplitudes(num_latent + num_trash, reps=5), range(0, num_latent + num_trash), inplace=True)
    circuit.barrier()

    # Swap‑test using auxiliary qubit
    aux = num_latent + 2 * num_trash
    circuit.h(aux)
    for i in range(num_trash):
        circuit.cswap(aux, num_latent + i, num_latent + num_trash + i)
    circuit.h(aux)
    circuit.measure(aux, cr[0])
    return circuit


class QuantumAutoencoder:
    """Sampler‑based autoencoder returning a probability vector."""
    def __init__(self, latent_dim: int = 3, trash_dim: int = 2):
        self.latent_dim = latent_dim
        self.trash_dim = trash_dim
        self.circuit = _autoencoder_circuit(latent_dim, trash_dim)
        self.sampler = Sampler()
        self.qnn = SamplerQNN(
            circuit=self.circuit,
            input_params=[],
            weight_params=self.circuit.parameters,
            interpret=lambda x: np.array([float(v) for v in x.values()]),
            output_shape=2,
            sampler=self.sampler,
        )

    def encode(self, inputs: np.ndarray) -> np.ndarray:
        """Encode input data into a probability vector via the sampler."""
        # For illustration, we ignore the embedding and run the circuit with fixed parameters
        return self.qnn.predict(inputs)

    def decode(self, latents: np.ndarray) -> np.ndarray:
        """Placeholder: decode is identity for illustration."""
        return latents


# --------------------------- Quantum Self‑Attention ---------------------------

class QuantumSelfAttention:
    """Self‑attention block implemented as a parameterized circuit."""
    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")
        self.backend = Aer.get_backend("qasm_simulator")

    def _build(self, rotation_params: np.ndarray, entangle_params: np.ndarray) -> QuantumCircuit:
        circuit = QuantumCircuit(self.qr, self.cr)
        for i in range(self.n_qubits):
            circuit.rx(rotation_params[3 * i], i)
            circuit.ry(rotation_params[3 * i + 1], i)
            circuit.rz(rotation_params[3 * i + 2], i)
        for i in range(self.n_qubits - 1):
            circuit.crx(entangle_params[i], i, i + 1)
        circuit.measure(self.qr, self.cr)
        return circuit

    def run(self, inputs: np.ndarray, rotation_params: np.ndarray, entangle_params: np.ndarray, shots: int = 1024) -> np.ndarray:
        circuit = self._build(rotation_params, entangle_params)
        job = self.backend.run(circuit, shots=shots)
        return np.array(list(job.result().get_counts(circuit).values()))


# --------------------------- Estimator QNN ---------------------------

def _create_estimator_qnn() -> EstimatorQNN:
    """Return a simple estimator QNN from the reference."""
    params = [Parameter("input"), Parameter("weight")]
    qc = QuantumCircuit(1)
    qc.h(0)
    qc.ry(params[0], 0)
    qc.rx(params[1], 0)
    observable = SparsePauliOp.from_list([("Y", 1)])
    estimator = Estimator()
    return EstimatorQNN(
        circuit=qc,
        observables=observable,
        input_params=[params[0]],
        weight_params=[params[1]],
        estimator=estimator,
    )


# --------------------------- Hybrid Model ---------------------------

class HybridAutoencoder:
    """Combined quantum autoencoder, self‑attention, and estimator."""
    def __init__(self, latent_dim: int = 3, trash_dim: int = 2, attention_qubits: int = 4):
        self.autoencoder = QuantumAutoencoder(latent_dim, trash_dim)
        self.attention = QuantumSelfAttention(attention_qubits)
        self.estimator = _create_estimator_qnn()

    def encode(self, inputs: np.ndarray) -> np.ndarray:
        return self.autoencoder.encode(inputs)

    def decode(self, latents: np.ndarray) -> np.ndarray:
        return self.autoencoder.decode(latents)

    def classify(self, inputs: np.ndarray) -> np.ndarray:
        return self.estimator.predict(inputs)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """End‑to‑end forward: encode → attention → decode."""
        z = self.encode(inputs)
        # Attention operates on classical vectors; here we convert to circuit counts
        rotation = np.random.rand(3 * self.attention.n_qubits)
        entangle = np.random.rand(self.attention.n_qubits - 1)
        attn_out = self.attention.run(z, rotation, entangle)
        return self.decode(attn_out)

    def train(self, data: np.ndarray, labels: np.ndarray | None = None, *, epochs: int = 20):
        """Very light‑weight training loop using COBYLA on the autoencoder weights."""
        opt = COBYLA(maxiter=200)
        for _ in range(epochs):
            # Encode data
            z = self.autoencoder.encode(data)
            # Reconstruction loss: MSE between sampled probs and input
            loss = np.mean((z - data) ** 2)
            # Optimize autoencoder parameters
            def func(params):
                for p, val in zip(self.autoencoder.circuit.parameters, params):
                    p.set_val(val)
                z = self.autoencoder.encode(data)
                return np.mean((z - data) ** 2)
            opt.optimize(len(self.autoencoder.circuit.parameters), func)
        # Estimator training omitted for brevity

__all__ = ["HybridAutoencoder"]
