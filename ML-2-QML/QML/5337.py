from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap, RealAmplitudes
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorSampler as Sampler, StatevectorEstimator as Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN, SamplerQNN
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from typing import Tuple, List

# ----------------------------------------------------------------------
# Quantum QCNN – compact variational ansatz with feature map
# ----------------------------------------------------------------------
def QCNN(num_qubits: int = 8) -> EstimatorQNN:
    estimator = Estimator()
    feature_map = ZFeatureMap(num_qubits)

    def conv_circuit(params):
        qc = QuantumCircuit(num_qubits)
        for i in range(num_qubits // 2):
            qc.cx(2 * i, 2 * i + 1)
            qc.ry(params[3 * i], 2 * i)
            qc.rz(params[3 * i + 1], 2 * i + 1)
            qc.cx(2 * i, 2 * i + 1)
        return qc

    def pool_circuit(params):
        qc = QuantumCircuit(num_qubits)
        for i in range(num_qubits // 2):
            qc.cx(2 * i, 2 * i + 1)
            qc.ry(params[3 * i], 2 * i)
            qc.rz(params[3 * i + 1], 2 * i + 1)
        return qc

    ansatz = QuantumCircuit(num_qubits)
    ansatz.append(conv_circuit(ParameterVector("c1", num_qubits // 2 * 3)), range(num_qubits), inplace=True)
    ansatz.append(pool_circuit(ParameterVector("p1", num_qubits // 2 * 3)), range(num_qubits), inplace=True)

    circuit = QuantumCircuit(num_qubits)
    circuit.append(feature_map, range(num_qubits), inplace=True)
    circuit.append(ansatz, range(num_qubits), inplace=True)

    observable = SparsePauliOp.from_list([("Z" + "I" * (num_qubits - 1), 1)])
    return EstimatorQNN(
        circuit=circuit.decompose(),
        observables=observable,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        estimator=estimator,
    )

# ----------------------------------------------------------------------
# Quantum classifier ansatz
# ----------------------------------------------------------------------
def build_classifier_circuit(num_qubits: int, depth: int) -> Tuple[QuantumCircuit, List, List, List[SparsePauliOp]]:
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    qc = QuantumCircuit(num_qubits)
    for q, p in zip(range(num_qubits), encoding):
        qc.rx(p, q)

    idx = 0
    for _ in range(depth):
        for q in range(num_qubits):
            qc.ry(weights[idx], q)
            idx += 1
        for q in range(num_qubits - 1):
            qc.cz(q, q + 1)

    observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1)) for i in range(num_qubits)]
    return qc, list(encoding), list(weights), observables

# ----------------------------------------------------------------------
# Quantum autoencoder via SamplerQNN
# ----------------------------------------------------------------------
def Autoencoder(num_latent: int = 3, num_trash: int = 2) -> SamplerQNN:
    sampler = Sampler()

    def ansatz(num_qubits):
        return RealAmplitudes(num_qubits, reps=5)

    def auto_encoder_circuit(num_latent, num_trash):
        qr = QuantumRegister(num_latent + 2 * num_trash + 1, "q")
        cr = ClassicalRegister(1, "c")
        qc = QuantumCircuit(qr, cr)
        qc.append(ansatz(num_latent + num_trash), range(0, num_latent + num_trash), inplace=True)
        qc.barrier()
        aux = num_latent + 2 * num_trash
        qc.h(aux)
        for i in range(num_trash):
            qc.cswap(aux, num_latent + i, num_latent + num_trash + i)
        qc.h(aux)
        qc.measure(aux, cr[0])
        return qc

    circuit = auto_encoder_circuit(num_latent, num_trash)
    return SamplerQNN(circuit=circuit, input_params=[], weight_params=circuit.parameters, sampler=sampler)

# ----------------------------------------------------------------------
# Quantum sampler circuit
# ----------------------------------------------------------------------
def SamplerQNN() -> SamplerQNN:
    inputs = ParameterVector("input", 2)
    weights = ParameterVector("weight", 4)

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
    return SamplerQNN(circuit=qc, input_params=inputs, weight_params=weights, sampler=sampler)

# ----------------------------------------------------------------------
# Wrapper that aggregates all quantum components
# ----------------------------------------------------------------------
class HybridQCNN:
    """
    Aggregates the QCNN, classifier, autoencoder and sampler QNNs into a single
    interface that can be used within a hybrid training loop.
    """

    def __init__(self, num_qubits: int = 8, depth: int = 3,
                 num_latent: int = 3, num_trash: int = 2):
        self.qcnn_qnn = QCNN(num_qubits)
        self.classifier_qnn, _, _, _ = build_classifier_circuit(num_qubits, depth)
        self.autoencoder_qnn = Autoencoder(num_latent, num_trash)
        self.sampler_qnn = SamplerQNN()

    def get_qnns(self) -> dict:
        return {
            "qcnn": self.qcnn_qnn,
            "classifier": self.classifier_qnn,
            "autoencoder": self.autoencoder_qnn,
            "sampler": self.sampler_qnn,
        }

__all__ = ["HybridQCNN", "QCNN", "build_classifier_circuit", "Autoencoder", "SamplerQNN"]
