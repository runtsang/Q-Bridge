import numpy as np
import networkx as nx
import itertools
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator as StatevectorEstimator
from qiskit_machine_learning.neural_networks import EstimatorQNN, SamplerQNN as QiskitSamplerQNN

# ----------------------------------------------------------------------
# Convolution and pooling primitives – inspired by the original QCNN
# ----------------------------------------------------------------------
def _conv_circuit(params):
    """Two‑qubit convolution unitary with 3 trainable angles."""
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

def _pool_circuit(params):
    """Two‑qubit pooling unitary with 3 trainable angles."""
    qc = QuantumCircuit(2)
    qc.rz(-np.pi / 2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    return qc

def conv_layer(num_qubits, param_prefix):
    """Assemble a convolution layer that applies _conv_circuit to every even‑odd pair."""
    qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
    for q1, q2 in zip(range(0, num_qubits, 2), range(1, num_qubits, 2)):
        sub = _conv_circuit(params[param_index:param_index+3])
        qc.append(sub, [q1, q2])
        param_index += 3
    return qc

def pool_layer(sources, sinks, param_prefix):
    """Apply _pool_circuit to pairs defined by sources and sinks."""
    num_qubits = len(sources) + len(sinks)
    qc = QuantumCircuit(num_qubits, name="Pooling Layer")
    param_index = 0
    params = ParameterVector(param_prefix, length=len(sources) * 3)
    for src, snk in zip(sources, sinks):
        sub = _pool_circuit(params[param_index:param_index+3])
        qc.append(sub, [src, snk])
        param_index += 3
    return qc

# ----------------------------------------------------------------------
# Fidelity‑based adjacency for quantum state vectors
# ----------------------------------------------------------------------
def fidelity_adjacency(states, threshold, *, secondary=None, secondary_weight=0.5):
    """
    Construct a weighted graph from pairwise fidelities of pure state vectors.
    `states` is a list/array of shape (N, 2**n) containing state amplitudes.
    """
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, a), (j, b) in itertools.combinations(enumerate(states), 2):
        fid = abs(np.vdot(a, b)) ** 2
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph

# ----------------------------------------------------------------------
# Main QCNN‑style variational circuit for regression
# ----------------------------------------------------------------------
def QCNNGen504QNN():
    """
    Return an EstimatorQNN that implements a QCNN with a regression ansatz.
    """
    # Feature map
    feature_map = ZFeatureMap(8)
    # Ansatz construction
    ansatz = QuantumCircuit(8, name="Ansatz")
    # First convolution / pooling pair
    ansatz.compose(conv_layer(8, "c1"), inplace=True)
    ansatz.compose(pool_layer([0,1,2,3], [4,5,6,7], "p1"), inplace=True)
    # Second convolution / pooling pair
    ansatz.compose(conv_layer(4, "c2"), inplace=True)
    ansatz.compose(pool_layer([0,1], [2,3], "p2"), inplace=True)
    # Third convolution / pooling pair
    ansatz.compose(conv_layer(2, "c3"), inplace=True)
    ansatz.compose(pool_layer([0], [1], "p3"), inplace=True)

    # Combine feature map and ansatz
    circuit = QuantumCircuit(8)
    circuit.compose(feature_map, inplace=True)
    circuit.compose(ansatz, inplace=True)

    # Observable for regression (Z on first qubit)
    observable = SparsePauliOp.from_list([("Z" + "I" * 7, 1)])

    # Estimator
    estimator = StatevectorEstimator()

    # Build QNN
    qnn = EstimatorQNN(
        circuit=circuit.decompose(),
        observables=observable,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        estimator=estimator,
    )
    return qnn

# ----------------------------------------------------------------------
# Sampler QNN for categorical output
# ----------------------------------------------------------------------
def QCNNGen504SamplerQNN():
    """
    Return a Qiskit SamplerQNN that produces a probability distribution
    over two outcomes from a parameterised 2‑qubit circuit.
    """
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

    sampler = QiskitSamplerQNN(
        circuit=qc,
        input_params=inputs,
        weight_params=weights,
    )
    return sampler

__all__ = ["QCNNGen504QNN", "QCNNGen504SamplerQNN", "fidelity_adjacency"]
