"""Quantum hybrid self‑attention classifier with optional regression.

The class exposes a quantum‑centric interface that mirrors the
classical HybridSelfAttention module.  The attention block is a
parameterised circuit that encodes the inputs via Ry rotations
followed by controlled‑X entangling gates.  The classifier block
is a layered ansatz with Pauli‑Z observables.  A small EstimatorQNN
is bundled for regression tasks.
"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple, Iterable

from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import StatevectorEstimator


def build_attention_circuit(n_qubits: int,
                            rotation_params: np.ndarray,
                            entangle_params: np.ndarray) -> QuantumCircuit:
    """Construct a small attention‑style circuit that mimics the
    classical self‑attention weight computation.
    """
    qc = QuantumCircuit(n_qubits)
    # Encode input via Ry rotations
    for i, angle in enumerate(rotation_params[:n_qubits]):
        qc.ry(angle, i)
    # Entangle neighbouring qubits
    for i, angle in enumerate(entangle_params[:n_qubits - 1]):
        qc.crx(angle, i, i + 1)
    # Measure all qubits
    qc.measure_all()
    return qc


def build_classifier_circuit(num_qubits: int,
                             depth: int) -> Tuple[QuantumCircuit,
                                                  Iterable,
                                                  Iterable,
                                                  List[SparsePauliOp]]:
    """Layered ansatz with encoding and variational parameters."""
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    qc = QuantumCircuit(num_qubits)
    for i, param in enumerate(encoding):
        qc.rx(param, i)

    idx = 0
    for _ in range(depth):
        for i in range(num_qubits):
            qc.ry(weights[idx], i)
            idx += 1
        for i in range(num_qubits - 1):
            qc.cz(i, i + 1)

    observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
                   for i in range(num_qubits)]
    return qc, list(encoding), list(weights), observables


class HybridQuantumSelfAttentionClassifier:
    """Quantum implementation of the hybrid attention + classifier."""
    def __init__(self,
                 n_qubits: int = 4,
                 depth: int = 2,
                 num_classes: int = 10) -> None:
        self.n_qubits = n_qubits
        self.depth = depth
        self.num_classes = num_classes
        self.backend = Aer.get_backend("statevector_simulator")

        # Placeholder circuits that will be parameterised later
        self.attn_circuit: QuantumCircuit | None = None
        self.classifier_circuit, self.enc_params, self.var_params, self.observables = \
            build_classifier_circuit(n_qubits, depth)

    def run_attention(self,
                      rotation_params: np.ndarray,
                      entangle_params: np.ndarray,
                      shots: int = 1024) -> dict:
        """Execute the attention circuit and return a probability distribution."""
        self.attn_circuit = build_attention_circuit(self.n_qubits,
                                                    rotation_params,
                                                    entangle_params)
        job = execute(self.attn_circuit, self.backend, shots=shots)
        return job.result().get_counts(self.attn_circuit)

    def run_classifier(self,
                       params: List[float],
                       shots: int = 1024) -> np.ndarray:
        """Given a list of variational parameters, evaluate the classifier
        expectation values for each Pauli‑Z observable.
        """
        # Map parameters to the circuit's symbolic parameters
        bound = dict(zip(self.var_params, params))
        bound.update(dict(zip(self.enc_params, params[:len(self.enc_params)])))
        qc = self.classifier_circuit.bind_parameters(bound)
        job = execute(qc, self.backend, shots=shots)
        result = job.result()
        # Convert counts to expectation values
        counts = result.get_counts(qc)
        probs = np.array([counts.get(bitstr, 0) for bitstr in result.get_counts(qc).keys()]) / shots
        return probs

    def estimator(self,
                  input_param: float,
                  weight_param: float) -> float:
        """Wrap the EstimatorQNN example for a single‑qubit regression."""
        params = [input_param, weight_param]
        estimator = StatevectorEstimator()
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.ry(params[0], 0)
        qc.rx(params[1], 0)
        observable = SparsePauliOp.from_list([("Y", 1)])
        est = EstimatorQNN(circuit=qc,
                           observables=observable,
                           input_params=[params[0]],
                           weight_params=[params[1]],
                           estimator=estimator)
        return est.predict(params)

__all__ = ["HybridQuantumSelfAttentionClassifier"]
