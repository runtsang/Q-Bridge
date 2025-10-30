"""Hybrid quantum autoencoder that mirrors QCNN convolution‑pooling and a swap‑test reconstruction."""

import json
import warnings
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap, RealAmplitudes
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.utils import algorithm_globals
from qiskit_machine_learning.neural_networks import SamplerQNN

# --------------------------------------------------------------------------- #
# Helper functions: QCNN convolution and pooling
# --------------------------------------------------------------------------- #
def _conv_circuit(params: ParameterVector) -> QuantumCircuit:
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

def _conv_layer(num_qubits: int, param_prefix: str) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
    qubits = list(range(num_qubits))
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits * 3)
    for q1, q2 in zip(qubits[0::2], qubits[1::2]):
        qc.compose(_conv_circuit(params[param_index : param_index + 3]), [q1, q2], inplace=True)
        qc.barrier()
        param_index += 3
    for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
        qc.compose(_conv_circuit(params[param_index : param_index + 3]), [q1, q2], inplace=True)
        qc.barrier()
        param_index += 3
    return qc

def _pool_circuit(params: ParameterVector) -> QuantumCircuit:
    qc = QuantumCircuit(2)
    qc.rz(-np.pi / 2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    return qc

def _pool_layer(sources: list[int], sinks: list[int], param_prefix: str) -> QuantumCircuit:
    num_qubits = len(sources) + len(sinks)
    qc = QuantumCircuit(num_qubits, name="Pooling Layer")
    param_index = 0
    params = ParameterVector(param_prefix, length=(num_qubits // 2) * 3)
    for src, sink in zip(sources, sinks):
        qc.compose(_pool_circuit(params[param_index : param_index + 3]), [src, sink], inplace=True)
        qc.barrier()
        param_index += 3
    return qc

# --------------------------------------------------------------------------- #
# Quantum autoencoder circuit
# --------------------------------------------------------------------------- #
def _auto_encoder_circuit(num_latent: int, num_trash: int) -> QuantumCircuit:
    qr = QuantumRegister(num_latent + 2 * num_trash + 1, "q")
    cr = ClassicalRegister(1, "c")
    circuit = QuantumCircuit(qr, cr)

    # Encoder: QCNN‑style conv‑pool layers
    ansatz = QuantumCircuit(num_latent + 2 * num_trash, name="Ansatz")
    ansatz.compose(_conv_layer(num_latent + num_trash, "c1"), range(num_latent + num_trash), inplace=True)
    ansatz.compose(_pool_layer(list(range(num_latent + num_trash)), list(range(num_latent + num_trash, num_latent + 2 * num_trash)), "p1"),
                   range(num_latent + 2 * num_trash), inplace=True)

    circuit.compose(ansatz, range(num_latent + 2 * num_trash), inplace=True)

    # Swap‑test reconstruction
    auxiliary = num_latent + 2 * num_trash
    circuit.h(auxiliary)
    for i in range(num_trash):
        circuit.cswap(auxiliary, num_latent + i, num_latent + num_trash + i)
    circuit.h(auxiliary)
    circuit.measure(auxiliary, cr[0])
    return circuit

# --------------------------------------------------------------------------- #
# Hybrid quantum autoencoder factory
# --------------------------------------------------------------------------- #
def HybridQuantumAutoencoder(
    num_latent: int = 3,
    num_trash: int = 2,
    feature_map_qubits: int = 8,
) -> SamplerQNN:
    """Return a :class:`~qiskit_machine_learning.neural_networks.SamplerQNN` that
    implements a QCNN‑style quantum autoencoder with a swap‑test readout.
    """
    algorithm_globals.random_seed = 42
    sampler = Sampler()

    # Feature map
    feature_map = ZFeatureMap(feature_map_qubits)
    feature_map.decompose()

    # Build ansatz with conv‑pool layers
    ansatz = QuantumCircuit(feature_map_qubits, name="Ansatz")
    ansatz.compose(_conv_layer(feature_map_qubits, "c1"), range(feature_map_qubits), inplace=True)
    ansatz.compose(_pool_layer(list(range(feature_map_qubits // 2)),
                               list(range(feature_map_qubits // 2, feature_map_qubits)),
                               "p1"),
                   range(feature_map_qubits), inplace=True)

    # Combine feature map and ansatz
    circuit = QuantumCircuit(feature_map_qubits)
    circuit.compose(feature_map, range(feature_map_qubits), inplace=True)
    circuit.compose(ansatz, range(feature_map_qubits), inplace=True)

    # Append autoencoder sub‑circuit
    ae_circ = _auto_encoder_circuit(num_latent, num_trash)
    circuit.compose(ae_circ, range(ae_circ.num_qubits), inplace=True)

    # Interpret as probability of measuring |1> on auxiliary qubit
    def interpret(x: np.ndarray) -> np.ndarray:
        return x[:, 0]  # probability of measuring 1

    qnn = SamplerQNN(
        circuit=circuit.decompose(),
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        interpret=interpret,
        output_shape=1,
        sampler=sampler,
    )
    return qnn


__all__ = ["HybridQuantumAutoencoder"]
