"""HybridQCNN – quantum implementation

This module constructs a variational circuit that mirrors the classical
convolutional stages, augments them with a quantum self‑attention block,
and produces a probability distribution via a SamplerQNN.  All components
are built using Qiskit and the Qiskit Machine Learning primitives.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN, SamplerQNN
from qiskit.primitives import StatevectorEstimator, StatevectorSampler
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.utils import algorithm_globals

# ----------------------------------------------------------------------
# 1.  Quantum convolution & pooling layers
# ----------------------------------------------------------------------
def conv_circuit(params):
    """Two‑qubit convolution unitary from the original QCNN paper."""
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

def pool_circuit(params):
    """Two‑qubit pooling unitary."""
    qc = QuantumCircuit(2)
    qc.rz(-np.pi / 2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    return qc

def conv_layer(num_qubits, param_prefix):
    """Wraps conv_circuit into a layer acting on adjacent qubit pairs."""
    qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
    params = ParameterVector(param_prefix, length=num_qubits * 3)
    for i in range(0, num_qubits, 2):
        sub = conv_circuit(params[i*3:(i+2)*3])
        qc.append(sub, [i, i+1])
    return qc

def pool_layer(num_qubits, param_prefix):
    """Wraps pool_circuit into a pooling layer."""
    qc = QuantumCircuit(num_qubits, name="Pooling Layer")
    params = ParameterVector(param_prefix, length=(num_qubits // 2) * 3)
    for i in range(0, num_qubits, 2):
        sub = pool_circuit(params[(i//2)*3:(i//2+1)*3])
        qc.append(sub, [i, i+1])
    return qc

# ----------------------------------------------------------------------
# 2.  Quantum self‑attention block
# ----------------------------------------------------------------------
def attention_block(n_qubits, param_prefix):
    """
    Simple quantum self‑attention module:
    - Rotations encode query/key/value parameters
    - Controlled‑RX gates entangle adjacent qubits
    """
    qc = QuantumCircuit(n_qubits, name="Quantum Self‑Attention")
    params = ParameterVector(param_prefix, length=n_qubits * 3 + (n_qubits-1))
    # Rotation layers
    for i in range(n_qubits):
        qc.rx(params[i*3], i)
        qc.ry(params[i*3+1], i)
        qc.rz(params[i*3+2], i)
    # Entangling layer
    for i in range(n_qubits-1):
        qc.crx(params[n_qubits*3 + i], i, i+1)
    # Measure only for debugging (classical readout not used in QNN)
    qc.measure_all()
    return qc

# ----------------------------------------------------------------------
# 3.  Full hybrid circuit
# ----------------------------------------------------------------------
def build_hybrid_circuit():
    """Construct the complete QCNN‑style circuit with an attention block."""
    feature_map = ZFeatureMap(8)          # 8‑qubit feature map
    ansatz = QuantumCircuit(8, name="Ansatz")

    # Convolution‑pool layers as in the original QCNN
    ansatz.compose(conv_layer(8, "c1"), inplace=True)
    ansatz.compose(pool_layer(8, "p1"), inplace=True)
    ansatz.compose(conv_layer(4, "c2"), inplace=True)
    ansatz.compose(pool_layer(4, "p2"), inplace=True)
    ansatz.compose(conv_layer(2, "c3"), inplace=True)
    ansatz.compose(pool_layer(2, "p3"), inplace=True)

    # Attach a small attention sub‑circuit on the first 4 qubits
    ansatz.compose(attention_block(4, "attn"), inplace=True)

    # Combine feature map and ansatz
    circuit = QuantumCircuit(8)
    circuit.compose(feature_map, inplace=True)
    circuit.compose(ansatz, inplace=True)
    return circuit

# ----------------------------------------------------------------------
# 4.  Quantum kernel (for optional use)
# ----------------------------------------------------------------------
def build_quantum_kernel():
    """
    Builds a simple quantum kernel using a fixed ansatz.
    Not used directly in the hybrid QNN but provided for completeness.
    """
    kernel_circuit = QuantumCircuit(4, name="Kernel")
    kernel_circuit.rz(np.pi / 2, 0)
    kernel_circuit.cx(0, 1)
    kernel_circuit.ry(np.pi / 4, 1)
    return kernel_circuit

# ----------------------------------------------------------------------
# 5.  HybridQCNN class
# ----------------------------------------------------------------------
class HybridQCNN:
    """
    Quantum version of HybridQCNN.

    The class exposes two interfaces:
    - `get_qnn()` returns an EstimatorQNN that can be used as a differentiable layer.
    - `get_sampler()` returns a SamplerQNN that outputs probability distributions.
    """

    def __init__(self) -> None:
        algorithm_globals.random_seed = 42
        self.estimator = StatevectorEstimator()
        self.sampler = StatevectorSampler()

        # Build the circuit
        self.circuit = build_hybrid_circuit().decompose()

        # Define observables and parameter lists
        self.input_params = [p for p in self.circuit.parameters if "θ" in str(p)]
        self.weight_params = [p for p in self.circuit.parameters if "c" in str(p) or "p" in str(p) or "attn" in str(p)]

        # Observable for a binary classifier (Z on qubit 0)
        self.observable = SparsePauliOp.from_list([("Z" + "I" * 7, 1)])

    def get_qnn(self) -> EstimatorQNN:
        """Return a variational QNN that outputs a single expectation value."""
        return EstimatorQNN(
            circuit=self.circuit,
            observables=self.observable,
            input_params=self.input_params,
            weight_params=self.weight_params,
            estimator=self.estimator,
        )

    def get_sampler(self) -> SamplerQNN:
        """Return a sampler that yields a probability distribution over 2 outcomes."""
        return SamplerQNN(
            circuit=self.circuit,
            input_params=self.input_params,
            weight_params=self.weight_params,
            sampler=self.sampler,
        )

def HybridQCNNFactory() -> HybridQCNN:
    """Convenience factory returning a ready‑to‑use instance."""
    return HybridQCNN()

__all__ = ["HybridQCNN", "HybridQCNNFactory", "build_hybrid_circuit", "build_quantum_kernel"]
