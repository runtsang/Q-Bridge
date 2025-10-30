"""Quantum hybrid autoencoder with attention and fraud‑detection style ansatz and classifier."""

from __future__ import annotations

from typing import Iterable, Tuple

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import ParameterVector
from qiskit.providers.aer import AerSimulator
from qiskit.quantum_info import StatevectorSampler, SparsePauliOp
import numpy as np

class HybridAutoencoder:
    def __init__(self, num_qubits: int, latent_qubits: int, depth: int = 2) -> None:
        self.num_qubits = num_qubits
        self.latent_qubits = latent_qubits
        self.depth = depth
        self.sim = AerSimulator()
        self.sampler = StatevectorSampler()
        self.circuit, self.encoding_params, self.weight_params, self.observables = self._build_circuit()

    def _build_circuit(self) -> Tuple[QuantumCircuit, Iterable, Iterable, list[SparsePauliOp]]:
        # Encoding
        encoding = ParameterVector("x", self.num_qubits)
        weights = ParameterVector("theta", self.num_qubits * self.depth)
        circuit = QuantumCircuit(self.num_qubits)
        for param, qubit in zip(encoding, range(self.num_qubits)):
            circuit.rx(param, qubit)

        # Self‑attention ansatz
        attn_params = ParameterVector("alpha", self.num_qubits * 3)
        for i in range(self.num_qubits):
            circuit.rx(attn_params[3 * i], i)
            circuit.ry(attn_params[3 * i + 1], i)
            circuit.rz(attn_params[3 * i + 2], i)
        for i in range(self.num_qubits - 1):
            circuit.crx(attn_params[self.num_qubits * 3 + i], i, i + 1)

        # Swap‑test to extract latent
        qr_latent = QuantumRegister(self.latent_qubits, "latent")
        circuit.add_register(qr_latent)
        aux = QuantumRegister(1, "aux")
        circuit.add_register(aux)
        circuit.h(aux[0])
        for i in range(self.latent_qubits):
            circuit.cswap(aux[0], i, qr_latent[i])
        circuit.h(aux[0])

        # Fraud‑detection style ansatz on latent qubits
        fraud_params = ParameterVector("beta", self.latent_qubits * 3)
        for i in range(self.latent_qubits):
            circuit.rx(fraud_params[3 * i], qr_latent[i])
            circuit.ry(fraud_params[3 * i + 1], qr_latent[i])
            circuit.rz(fraud_params[3 * i + 2], qr_latent[i])
        for i in range(self.latent_qubits - 1):
            circuit.cx(qr_latent[i], qr_latent[i + 1])

        # Decoding (inverse of encoding)
        for param, qubit in zip(reversed(encoding), range(self.num_qubits)):
            circuit.rx(-param, qubit)

        # Classifier observables
        obs = [SparsePauliOp("I" * i + "Z" + "I" * (self.num_qubits - i - 1)) for i in range(self.num_qubits)]
        return circuit, encoding, weights, obs

    def bind_parameters(self, param_dict: dict[str, float]) -> None:
        self.circuit.assign_parameters(param_dict, inplace=True)

    def run(self, shots: int = 1024, param_values: dict[str, float] | None = None) -> dict[str, np.ndarray]:
        if param_values:
            self.bind_parameters(param_values)
        job = self.sim.run(self.circuit, shots=shots)
        result = job.result()
        counts = result.get_counts(self.circuit)
        # Interpret counts as reconstruction (simple binary vector) and classification logits
        recon = np.array([int(state, 2) for state in counts.keys()]) / shots
        logits = np.array([result.get_expectation_value(obs) for obs in self.observables])
        return {"reconstruction": recon, "logits": logits}

def HybridAutoencoderFactory(num_qubits: int, latent_qubits: int, depth: int = 2) -> HybridAutoencoder:
    return HybridAutoencoder(num_qubits, latent_qubits, depth)

__all__ = ["HybridAutoencoder", "HybridAutoencoderFactory"]
