import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN

def conv_circuit(params: ParameterVector) -> QuantumCircuit:
    qc = QuantumCircuit(2)
    qc.rz(-np.pi/2, 1)
    qc.cx(1,0)
    qc.rz(params[0],0)
    qc.ry(params[1],1)
    qc.cx(0,1)
    qc.ry(params[2],1)
    qc.cx(1,0)
    qc.rz(np.pi/2,0)
    return qc

def pool_circuit(params: ParameterVector) -> QuantumCircuit:
    qc = QuantumCircuit(2)
    qc.rz(-np.pi/2, 1)
    qc.cx(1,0)
    qc.rz(params[0],0)
    qc.ry(params[1],1)
    qc.cx(0,1)
    qc.ry(params[2],1)
    return qc

def conv_layer(num_qubits: int, param_prefix: str) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits, name="Convolution Layer")
    params = ParameterVector(param_prefix, length=num_qubits//2*3)
    for i in range(0, num_qubits, 2):
        qc.append(conv_circuit(params[i//2*3:(i//2+1)*3]), [i, i+1])
    return qc

def pool_layer(sources: list[int], sinks: list[int], param_prefix: str) -> QuantumCircuit:
    num_qubits = len(sources)+len(sinks)
    qc = QuantumCircuit(num_qubits, name="Pooling Layer")
    params = ParameterVector(param_prefix, length=len(sources)*3)
    for idx, (src, snk) in enumerate(zip(sources, sinks)):
        qc.append(pool_circuit(params[idx*3:(idx+1)*3]), [src, snk])
    return qc

def build_qcnn_ansatz(num_qubits: int = 8, depth: int = 3) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits)
    for d in range(depth):
        qc.compose(conv_layer(num_qubits, f"c{d}"), inplace=True)
        if num_qubits > 1:
            sinks = list(range(num_qubits//2))
            sources = list(range(num_qubits//2, num_qubits))
            qc.compose(pool_layer(sources, sinks, f"p{d}"), inplace=True)
            num_qubits = len(sinks)
    return qc

def QCNNEnhancedQML(num_qubits: int = 8,
                    depth: int = 3,
                    noise_model=None) -> EstimatorQNN:
    """
    Construct a QCNN‑style variational quantum neural network.

    Parameters
    ----------
    num_qubits : int
        Total number of qubits used for the feature map and ansatz.
    depth : int
        Number of convolution‑pooling pairs in the ansatz.
    noise_model : qiskit.providers.models.NoiseModel | None
        Optional noise model to attach to the estimator.

    Returns
    -------
    EstimatorQNN
        Ready‑to‑use quantum neural network instance.
    """
    feature_map = ZFeatureMap(num_qubits)
    ansatz = build_qcnn_ansatz(num_qubits, depth)

    circuit = QuantumCircuit(num_qubits)
    circuit.compose(feature_map, inplace=True)
    circuit.compose(ansatz, inplace=True)

    observable = SparsePauliOp.from_list([("Z" + "I" * (num_qubits-1), 1)])

    estimator = Estimator(noise_model=noise_model)

    qnn = EstimatorQNN(circuit=circuit.decompose(),
                       observables=observable,
                       input_params=feature_map.parameters,
                       weight_params=ansatz.parameters,
                       estimator=estimator)
    return qnn

__all__ = ["QCNNEnhancedQML"]
