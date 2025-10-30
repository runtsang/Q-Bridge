import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap, RealAmplitudes
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.utils import algorithm_globals

def _conv_circuit(params):
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

def conv_layer(num_qubits: int, param_prefix: str) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
    params = ParameterVector(param_prefix, length=num_qubits*3)
    param_index = 0
    for i in range(0, num_qubits, 2):
        sub = _conv_circuit(params[param_index:param_index+3])
        qc.append(sub, [i, i+1])
        qc.barrier()
        param_index += 3
    return qc

def _pool_circuit(params):
    qc = QuantumCircuit(2)
    qc.rz(-np.pi/2, 1)
    qc.cx(1,0)
    qc.rz(params[0],0)
    qc.ry(params[1],1)
    qc.cx(0,1)
    qc.ry(params[2],1)
    return qc

def pool_layer(sources, sinks, param_prefix: str) -> QuantumCircuit:
    qc = QuantumCircuit(len(sources)+len(sinks), name="Pooling Layer")
    params = ParameterVector(param_prefix, length=len(sources)*3)
    param_index = 0
    for src, snk in zip(sources, sinks):
        sub = _pool_circuit(params[param_index:param_index+3])
        qc.append(sub, [src, snk])
        qc.barrier()
        param_index += 3
    return qc

def _autoencoder_ansatz(num_latent: int, num_trash: int) -> QuantumCircuit:
    qr = QuantumCircuit(num_latent + 2*num_trash + 1, name="Autoencoder Ansatz")
    ra = RealAmplitudes(num_latent + num_trash, reps=5)
    qr.append(ra, list(range(num_latent + num_trash)))
    qr.barrier()
    aux = num_latent + 2*num_trash
    qr.h(aux)
    for i in range(num_trash):
        qr.cswap(aux, num_latent + i, num_latent + num_trash + i)
    qr.h(aux)
    qr.measure(aux, 0)
    return qr

def HybridQCNNAutoencoderQNN(
    num_latent: int = 3,
    num_trash: int = 2,
) -> EstimatorQNN:
    """
    Constructs a quantum neural network that merges a QCNN ansatz with a quantum autoencoder.
    """
    algorithm_globals.random_seed = 42
    feature_map = ZFeatureMap(8)

    # QCNN ansatz
    qc = QuantumCircuit(8)
    qc.compose(conv_layer(8, "c1"), inplace=True)
    qc.compose(pool_layer([0,1,2,3], [4,5,6,7], "p1"), inplace=True)
    qc.compose(conv_layer(4, "c2"), inplace=True)
    qc.compose(pool_layer([0,1], [2,3], "p2"), inplace=True)
    qc.compose(conv_layer(2, "c3"), inplace=True)
    qc.compose(pool_layer([0], [1], "p3"), inplace=True)

    # Autoencoder ansatz
    ae = _autoencoder_ansatz(num_latent, num_trash)

    # Build full circuit
    total_qubits = 8 + num_latent + 2*num_trash + 1
    full = QuantumCircuit(total_qubits)
    full.compose(feature_map, range(8), inplace=True)
    full.compose(qc, range(8), inplace=True)
    full.compose(ae, range(8, total_qubits), inplace=True)

    # Observable acting on the first qubit of the autoencoder output
    obs = SparsePauliOp.from_list([("Z" + "I"*(total_qubits-1), 1)])

    estimator = Estimator()
    qnn = EstimatorQNN(
        circuit=full.decompose(),
        observables=obs,
        input_params=feature_map.parameters,
        weight_params=ae.parameters,
        estimator=estimator,
    )
    return qnn

__all__ = ["HybridQCNNAutoencoderQNN"]
