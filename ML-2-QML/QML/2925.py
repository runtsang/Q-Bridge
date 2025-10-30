import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit_machine_learning.utils import algorithm_globals
from qiskit_machine_learning.neural_networks import SamplerQNN

def _conv_layer(num_qubits: int, param_prefix: str) -> QuantumCircuit:
    """QCNN‑style convolutional layer composed of 2‑qubit blocks."""
    qc = QuantumCircuit(num_qubits)
    params = ParameterVector(param_prefix, length=num_qubits * 3)
    param_index = 0
    for q1, q2 in zip(range(0, num_qubits, 2), range(1, num_qubits, 2)):
        sub = QuantumCircuit(2)
        sub.rz(-np.pi / 2, 1)
        sub.cx(1, 0)
        sub.rz(params[param_index], 0)
        sub.ry(params[param_index + 1], 1)
        sub.cx(0, 1)
        sub.ry(params[param_index + 2], 1)
        sub.cx(1, 0)
        sub.rz(np.pi / 2, 0)
        qc.append(sub.to_instruction(), [q1, q2])
        param_index += 3
    return qc

def _pool_layer(num_qubits: int, param_prefix: str) -> QuantumCircuit:
    """QCNN‑style pooling layer that reduces the qubit count by half."""
    qc = QuantumCircuit(num_qubits)
    params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
    param_index = 0
    for q1, q2 in zip(range(0, num_qubits, 2), range(1, num_qubits, 2)):
        sub = QuantumCircuit(2)
        sub.rz(-np.pi / 2, 1)
        sub.cx(1, 0)
        sub.rz(params[param_index], 0)
        sub.ry(params[param_index + 1], 1)
        sub.cx(0, 1)
        sub.ry(params[param_index + 2], 1)
        qc.append(sub.to_instruction(), [q1, q2])
        param_index += 3
    return qc

def AutoencoderGen191() -> SamplerQNN:
    """Quantum auto‑encoder that mirrors the hybrid classical architecture."""
    algorithm_globals.random_seed = 42
    sampler = Sampler()

    # Feature map: encode classical data into a 8‑qubit Hilbert space
    feature_map = ZFeatureMap(num_qubits=8)
    num_qubits = 8

    # Build a QCNN ansatz: alternating conv and pool layers
    ansatz = QuantumCircuit(num_qubits, name="QCNN-Ansatz")
    ansatz.compose(_conv_layer(num_qubits, "c1"), range(num_qubits), inplace=True)
    ansatz.compose(_pool_layer(num_qubits, "p1"), range(num_qubits), inplace=True)
    ansatz.compose(_conv_layer(num_qubits // 2, "c2"), range(num_qubits // 2), inplace=True)
    ansatz.compose(_pool_layer(num_qubits // 2, "p2"), range(num_qubits // 2), inplace=True)
    ansatz.compose(_conv_layer(num_qubits // 4, "c3"), range(num_qubits // 4), inplace=True)
    ansatz.compose(_pool_layer(num_qubits // 4, "p3"), range(num_qubits // 4), inplace=True)

    # Swap‑test based reconstruction: ancilla qubit measures fidelity
    qc = QuantumCircuit(num_qubits + 1, 1, name="Autoencoder-Circuit")
    qc.compose(feature_map, range(num_qubits), inplace=True)
    qc.compose(ansatz, range(num_qubits), inplace=True)

    ancilla = num_qubits
    qc.h(ancilla)
    # Swap test between the first two latent qubits
    qc.cswap(ancilla, 0, 1)
    qc.h(ancilla)
    qc.measure(ancilla, 0)

    def identity(x):
        """Return raw measurement probabilities."""
        return x

    qnn = SamplerQNN(
        circuit=qc,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        interpret=identity,
        output_shape=2,
        sampler=sampler,
    )
    return qnn

__all__ = ["AutoencoderGen191"]
