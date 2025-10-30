"""HybridSamplerCNN: a quantum neural network that fuses a 2‑qubit sampler with a QCNN ansatz.

The quantum circuit first embeds the input data via an 8‑qubit ZFeatureMap,
then applies a parameterised 2‑qubit sampler (mirroring SamplerQNN),
followed by a hierarchical QCNN ansatz consisting of convolutional and pooling layers.
The EstimatorQNN wrapper exposes a `predict` method that accepts classical
feature vectors and returns the expectation value of the observable
Z⊗I⁷, which is interpreted as a binary classification score.
"""

from __future__ import annotations

import numpy as np
import torch

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning import algorithm_globals

# reproducible random seed
algorithm_globals.random_seed = 12345

# ----- Sampler sub‑circuit (2 qubits) -----
sampler_inputs = ParameterVector("sampler_input", 2)
sampler_weights = ParameterVector("sampler_weight", 4)

sampler_circuit = QuantumCircuit(2)
sampler_circuit.ry(sampler_inputs[0], 0)
sampler_circuit.ry(sampler_inputs[1], 1)
sampler_circuit.cx(0, 1)
sampler_circuit.ry(sampler_weights[0], 0)
sampler_circuit.ry(sampler_weights[1], 1)
sampler_circuit.cx(0, 1)
sampler_circuit.ry(sampler_weights[2], 0)
sampler_circuit.ry(sampler_weights[3], 1)

# ----- Convolution & pooling primitives -----
def conv_circuit(params: ParameterVector) -> QuantumCircuit:
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

def pool_circuit(params: ParameterVector) -> QuantumCircuit:
    qc = QuantumCircuit(2)
    qc.rz(-np.pi / 2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    return qc

def conv_layer(num_qubits: int, param_prefix: str) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
    qubits = list(range(num_qubits))
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
    for q1, q2 in zip(qubits[0::2], qubits[1::2]):
        qc.compose(conv_circuit(params[param_index : param_index + 3]), [q1, q2], inplace=True)
        qc.barrier()
        param_index += 3
    for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
        qc.compose(conv_circuit(params[param_index : param_index + 3]), [q1, q2], inplace=True)
        qc.barrier()
        param_index += 3
    return qc

def pool_layer(sources: list[int], sinks: list[int], param_prefix: str) -> QuantumCircuit:
    num_qubits = len(sources) + len(sinks)
    qc = QuantumCircuit(num_qubits, name="Pooling Layer")
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
    for source, sink in zip(sources, sinks):
        qc.compose(pool_circuit(params[param_index : param_index + 3]), [source, sink], inplace=True)
        qc.barrier()
        param_index += 3
    return qc

# ----- QCNN ansatz -----
ansatz = QuantumCircuit(8, name="QCNN Ansatz")
ansatz.compose(conv_layer(8, "c1"), list(range(8)), inplace=True)
ansatz.compose(pool_layer([0, 1, 2, 3], [4, 5, 6, 7], "p1"), list(range(8)), inplace=True)
ansatz.compose(conv_layer(4, "c2"), list(range(4, 8)), inplace=True)
ansatz.compose(pool_layer([0, 1], [2, 3], "p2"), list(range(4, 8)), inplace=True)
ansatz.compose(conv_layer(2, "c3"), list(range(6, 8)), inplace=True)
ansatz.compose(pool_layer([0], [1], "p3"), list(range(6, 8)), inplace=True)

# ----- Full hybrid circuit -----
circuit = QuantumCircuit(8, name="Hybrid Sampler‑QCNN")
# Insert sampler on qubits 0 and 1
circuit.compose(sampler_circuit, [0, 1], inplace=True)

# Feature map
feature_map = ZFeatureMap(8)
circuit.compose(feature_map, range(8), inplace=True)

# Attach QCNN ansatz
circuit.compose(ansatz, range(8), inplace=True)

# Observable for binary classification
observable = SparsePauliOp.from_list([("Z" + "I" * 7, 1)])

# Estimator backend
estimator = Estimator()

# ----- HybridSamplerCNN wrapper -----
class HybridSamplerCNN(EstimatorQNN):
    """
    Quantum neural network that fuses a 2‑qubit sampler with a QCNN ansatz.
    Inherits from EstimatorQNN and provides a `predict` method that accepts
    8‑dimensional feature vectors.
    """

    def __init__(self) -> None:
        # Input parameters: feature map + sampler inputs
        input_params = feature_map.parameters + sampler_inputs
        # Weight parameters: sampler weights + all QCNN circuit weights
        weight_params = sampler_weights + ansatz.parameters
        super().__init__(
            circuit=circuit.decompose(),
            observables=observable,
            input_params=input_params,
            weight_params=weight_params,
            estimator=estimator,
        )

    def predict(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """
        Accepts a torch tensor of shape (batch, 8) and returns a tensor of
        expectation values (batch, 1) that can be interpreted as class scores.
        """
        import numpy as np
        return super().predict(x.numpy())

__all__ = ["HybridSamplerCNN"]
