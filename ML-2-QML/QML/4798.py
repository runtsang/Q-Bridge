from __future__ import annotations

import numpy as np
from typing import Iterable, Tuple

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Sampler as SamplerPrimitive
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.utils import algorithm_globals

algorithm_globals.random_seed = 42
sampler = SamplerPrimitive()


def build_autoencoder_circuit(num_latent: int, num_trash: int) -> Tuple[QuantumCircuit, Iterable, Iterable]:
    """Variational auto‑encoder circuit (RealAmplitudes ansatz)."""
    num_qubits = num_latent + 2 * num_trash + 1
    qr = QuantumRegister(num_qubits, "q")
    cr = ClassicalRegister(1, "c")
    circuit = QuantumCircuit(qr, cr)

    # Encoder ansatz
    from qiskit.circuit.library import RealAmplitudes
    ansatz = RealAmplitudes(num_latent + num_trash, reps=5)
    circuit.compose(ansatz, range(0, num_latent + num_trash), inplace=True)

    # Swap‑test for latent extraction
    auxiliary = num_latent + 2 * num_trash
    circuit.h(auxiliary)
    for i in range(num_trash):
        circuit.cswap(auxiliary, num_latent + i, num_latent + num_trash + i)
    circuit.h(auxiliary)
    circuit.measure(auxiliary, cr[0])

    return circuit, ansatz.parameters, []


def build_attention_circuit(num_qubits: int) -> Tuple[QuantumCircuit, Iterable, Iterable]:
    """Quantum self‑attention block."""
    qr = QuantumRegister(num_qubits, "q")
    cr = ClassicalRegister(num_qubits, "c")
    circuit = QuantumCircuit(qr, cr)

    rot_params = ParameterVector("r", 3 * num_qubits)
    ent_params = ParameterVector("e", num_qubits - 1)

    for i in range(num_qubits):
        circuit.rx(rot_params[3 * i], i)
        circuit.ry(rot_params[3 * i + 1], i)
        circuit.rz(rot_params[3 * i + 2], i)

    for i in range(num_qubits - 1):
        circuit.crx(ent_params[i], i, i + 1)

    circuit.measure(qr, cr)
    return circuit, rot_params, ent_params


def build_classifier_circuit(num_qubits: int, depth: int) -> Tuple[QuantumCircuit, Iterable, Iterable, list[SparsePauliOp]]:
    """Variational classifier ansatz with explicit encoding."""
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    circuit = QuantumCircuit(num_qubits)
    for param, qubit in zip(encoding, range(num_qubits)):
        circuit.rx(param, qubit)

    idx = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            circuit.ry(weights[idx], qubit)
            idx += 1
        for qubit in range(num_qubits - 1):
            circuit.cz(qubit, qubit + 1)

    observables = [
        SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
        for i in range(num_qubits)
    ]
    return circuit, list(encoding), list(weights), observables


class HybridAutoEncoderQuantum:
    """Quantum counterpart of the hybrid auto‑encoder."""
    def __init__(self, num_latent: int = 3, num_trash: int = 2, num_qubits: int = 4, depth: int = 2):
        self.auto_circ, self.auto_params, _ = build_autoencoder_circuit(num_latent, num_trash)
        self.attn_circ, self.attn_rot, self.attn_ent = build_attention_circuit(num_qubits)
        self.classif_circ, self.classif_enc, self.classif_wts, self.classif_obs = build_classifier_circuit(num_qubits, depth)

        # Chain circuits
        self.combined_circuit = QuantumCircuit(self.auto_circ.num_qubits)
        self.combined_circuit.compose(self.auto_circ, inplace=True)
        # Note: For simplicity we ignore cross‑qubit interactions between blocks
        self.combined_circuit.compose(self.attn_circ, inplace=True)
        self.combined_circuit.compose(self.classif_circ, inplace=True)

        all_params = list(self.auto_params) + list(self.attn_rot) + list(self.attn_ent) + list(self.classif_enc) + list(self.classif_wts)
        self.qnn = SamplerQNN(
            circuit=self.combined_circuit,
            input_params=[],
            weight_params=all_params,
            interpret=lambda x: x,
            output_shape=len(self.classif_obs),
            sampler=sampler,
        )

    def run(
        self,
        params: dict[str, float],
        shots: int = 1024,
    ) -> np.ndarray:
        """Evaluate the combined circuit with given variational parameters."""
        return self.qnn.predict(params, shots=shots)


__all__ = ["HybridAutoEncoderQuantum", "build_autoencoder_circuit", "build_attention_circuit", "build_classifier_circuit"]
