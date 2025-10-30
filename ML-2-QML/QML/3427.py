import numpy as np
from qiskit import QuantumCircuit, ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator as Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.utils import algorithm_globals

def QCNNHybrid() -> EstimatorQNN:
    """
    Builds a quantum‑convolution‑based neural network that incorporates
    a random‑layer module inspired by Quantum‑NAT.  The ansatz consists of
    three convolutional layers followed by three pooling layers, and
    finally a parametrised random rotation block that injects additional
    expressivity.
    """
    algorithm_globals.random_seed = 12345
    estimator = Estimator()

    # ----- Basic convolution gate used in the QCNN layers -----
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

    # ----- Pooling gate (no measurement) -----
    def pool_circuit(params):
        qc = QuantumCircuit(2)
        qc.rz(-np.pi / 2, 1)
        qc.cx(1, 0)
        qc.rz(params[0], 0)
        qc.ry(params[1], 1)
        qc.cx(0, 1)
        qc.ry(params[2], 1)
        return qc

    # ----- Convolutional layer over n qubits -----
    def conv_layer(num_qubits, param_prefix):
        qc = QuantumCircuit(num_qubits, name="ConvLayer")
        params = ParameterVector(param_prefix, length=num_qubits * 3)
        for i in range(0, num_qubits, 2):
            qc.compose(conv_circuit(params[i*3:(i+1)*3]), [i, i+1], inplace=True)
        return qc

    # ----- Pooling layer with arbitrary source/sink mapping -----
    def pool_layer(sources, sinks, param_prefix):
        num_qubits = len(sources) + len(sinks)
        qc = QuantumCircuit(num_qubits, name="PoolLayer")
        params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
        for src, sink, p in zip(sources, sinks,
                               [params[i*3:(i+1)*3] for i in range(num_qubits // 2)]):
            qc.compose(pool_circuit(p), [src, sink], inplace=True)
        return qc

    # ----- Random‑layer inspired by Quantum‑NAT (all‑to‑all rotations) -----
    def random_layer(num_qubits, param_prefix):
        qc = QuantumCircuit(num_qubits, name="RandomLayer")
        params = ParameterVector(param_prefix, length=num_qubits * 3)
        for i in range(num_qubits):
            qc.rx(params[i*3], i)
            qc.ry(params[i*3+1], i)
            qc.rz(params[i*3+2], i)
        return qc

    # ---- Build the full ansatz ----
    ansatz = QuantumCircuit(8, name="HybridAnsatz")

    # First convolution & pooling
    ansatz.compose(conv_layer(8, "c1"), list(range(8)), inplace=True)
    ansatz.compose(pool_layer([0,1,2,3], [4,5,6,7], "p1"), list(range(8)), inplace=True)

    # Second convolution & pooling
    ansatz.compose(conv_layer(4, "c2"), list(range(4,8)), inplace=True)
    ansatz.compose(pool_layer([0,1], [2,3], "p2"), list(range(4,8)), inplace=True)

    # Third convolution & pooling
    ansatz.compose(conv_layer(2, "c3"), list(range(6,8)), inplace=True)
    ansatz.compose(pool_layer([0], [1], "p3"), list(range(6,8)), inplace=True)

    # Add the random‑layer block
    ansatz.compose(random_layer(8, "rl"), list(range(8)), inplace=True)

    # Feature map + ansatz composition
    feature_map = ZFeatureMap(8)
    circuit = QuantumCircuit(8)
    circuit.compose(feature_map, range(8), inplace=True)
    circuit.compose(ansatz, range(8), inplace=True)

    # Observable (single‑qubit Z on the first qubit)
    observable = SparsePauliOp.from_list([("Z" + "I" * 7, 1)])

    # Build the EstimatorQNN
    qnn = EstimatorQNN(
        circuit=circuit.decompose(),
        observables=observable,
        input_params=feature_map.parameters,
        weight_params=[p for p in ansatz.parameters if "c" in p or "p" in p or "rl" in p],
        estimator=estimator,
    )
    return qnn

class QCNNHybridModel:
    """
    Thin wrapper around the EstimatorQNN that mimics a PyTorch module.
    The __call__ method accepts a NumPy array or PyTorch tensor of shape
    [batch, 8] and returns the predicted outputs.
    """
    def __init__(self):
        self.qnn = QCNNHybrid()

    def __call__(self, inputs):
        if isinstance(inputs, torch.Tensor):
            inputs = inputs.detach().cpu().numpy()
        return self.qnn.predict(inputs)

__all__ = ["QCNNHybridModel", "QCNNHybrid"]
