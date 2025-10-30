from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Sampler
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.utils import algorithm_globals

# --------------------------------------------------------------------------- #
# Quantum autoencoder mirroring the QCNN ansatz
# --------------------------------------------------------------------------- #
def HybridQuantumAutoencoder() -> SamplerQNN:
    """
    Constructs a variational autoencoder circuit that uses a QCNN‑style
    convolution + pooling ansatz followed by a feature map.  The circuit
    is wrapped in a SamplerQNN so that the latent representation can be
    optimized with a classical optimiser.
    """
    algorithm_globals.random_seed = 42
    sampler = Sampler()

    # ----- Convolution block -----------------------------------------------
    def conv_circuit(params):
        qc = QuantumCircuit(2)
        qc.rz(-np.pi / 2, 1)
        qc.cx(1, 0)
        qc.rz(params[0], 0)
        qc.ry(params[1], 1)
        qc.cx(0, 1)
        qc.ry(params[2], 1)
        qc.cx(1, 0)
        qc.rz(np.pi / 2, 0)
        return qc

    # ----- Convolutional layer ---------------------------------------------
    def conv_layer(num_qubits, param_prefix):
        qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
        qubits = list(range(num_qubits))
        param_index = 0
        params = ParameterVector(param_prefix, length=num_qubits * 3)
        for q1, q2 in zip(qubits[0::2], qubits[1::2]):
            qc.append(conv_circuit(params[param_index : param_index + 3]), [q1, q2])
            qc.barrier()
            param_index += 3
        for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
            qc.append(conv_circuit(params[param_index : param_index + 3]), [q1, q2])
            qc.barrier()
            param_index += 3
        return qc

    # ----- Pooling block ----------------------------------------------------
    def pool_circuit(params):
        qc = QuantumCircuit(2)
        qc.rz(-np.pi / 2, 1)
        qc.cx(1, 0)
        qc.rz(params[0], 0)
        qc.ry(params[1], 1)
        qc.cx(0, 1)
        qc.ry(params[2], 1)
        return qc

    # ----- Pooling layer ----------------------------------------------------
    def pool_layer(sources, sinks, param_prefix):
        num_qubits = len(sources) + len(sinks)
        qc = QuantumCircuit(num_qubits, name="Pooling Layer")
        param_index = 0
        params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
        for source, sink in zip(sources, sinks):
            qc.append(pool_circuit(params[param_index : param_index + 3]), [source, sink])
            qc.barrier()
            param_index += 3
        return qc

    # -----------------------------------------------------------------------
    # Build the full ansatz with 3 convolution + pooling stages
    # -----------------------------------------------------------------------
    feature_map = ZFeatureMap(8)
    ansatz = QuantumCircuit(8)

    ansatz.compose(conv_layer(8, "c1"), range(8), inplace=True)
    ansatz.compose(pool_layer([0, 1, 2, 3], [4, 5, 6, 7], "p1"), range(8), inplace=True)
    ansatz.compose(conv_layer(4, "c2"), range(4, 8), inplace=True)
    ansatz.compose(pool_layer([0, 1], [2, 3], "p2"), range(4, 8), inplace=True)
    ansatz.compose(conv_layer(2, "c3"), range(6, 8), inplace=True)
    ansatz.compose(pool_layer([0], [1], "p3"), range(6, 8), inplace=True)

    # Combine feature map and ansatz into a single circuit
    circuit = QuantumCircuit(8)
    circuit.compose(feature_map, range(8), inplace=True)
    circuit.compose(ansatz, range(8), inplace=True)

    # Observable for a simple autoencoder loss (Z on first qubit)
    observable = SparsePauliOp.from_list([("Z" + "I" * 7, 1)])

    # Wrap in a SamplerQNN
    qnn = SamplerQNN(
        circuit=circuit.decompose(),
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        sampler=sampler,
        interpret=lambda x: x[0],  # return the first expectation value
        output_shape=1,
    )
    return qnn


__all__ = ["HybridQuantumAutoencoder"]
