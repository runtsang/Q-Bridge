import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import ZFeatureMap
from qiskit.primitives import Estimator
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.utils import algorithm_globals
from qiskit.circuit import ParameterVector

def conv_layer(num_qubits: int, param_prefix: str) -> QuantumCircuit:
    """Convolutional block: pairs of qubits are processed by a small
    parameterised two‑qubit unitary.  The parameters are grouped by
    ``param_prefix`` so that each block has its own trainable weights."""
    qc = QuantumCircuit(num_qubits)
    qubits = list(range(num_qubits))
    params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
    for i in range(0, len(qubits), 2):
        sub = QuantumCircuit(2)
        sub.rz(-np.pi / 2, 1)
        sub.cx(1, 0)
        sub.rz(params[i], 0)
        sub.ry(params[i + 1], 1)
        sub.cx(0, 1)
        sub.ry(params[i + 2], 1)
        sub.cx(1, 0)
        sub.rz(np.pi / 2, 0)
        qc.append(sub.to_instruction(), [qubits[i], qubits[i + 1]])
    return qc

def pool_layer(num_qubits: int, param_prefix: str) -> QuantumCircuit:
    """Pooling block that reduces the number of active qubits by
    entangling neighbouring pairs and discarding one qubit."""
    qc = QuantumCircuit(num_qubits)
    params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
    for i in range(0, len(params), 3):
        sub = QuantumCircuit(2)
        sub.rz(-np.pi / 2, 1)
        sub.cx(1, 0)
        sub.rz(params[i], 0)
        sub.ry(params[i + 1], 1)
        sub.cx(0, 1)
        sub.ry(params[i + 2], 1)
        qc.append(sub.to_instruction(), [i, i + 1])
    return qc

def domain_wall(qc: QuantumCircuit, start: int, end: int) -> QuantumCircuit:
    """Introduce a domain wall by flipping all qubits in the interval
    ``[start, end)``.  This is a simple way to inject structured noise
    into the circuit."""
    for i in range(start, end):
        qc.x(i)
    return qc

def identity_interpret(x):
    """Return the raw expectation value produced by EstimatorQNN."""
    return x

class HybridAutoencoder(EstimatorQNN):
    """Quantum autoencoder that combines a QCNN‑style ansatz with a
    swap‑test based latent reconstruction.  The circuit is built from
    a ZFeatureMap, a stack of convolution and pooling layers, and a
    domain‑wall perturbation before the swap‑test."""
    def __init__(self, num_latent: int = 3, num_trash: int = 2, seed: int = 42):
        algorithm_globals.random_seed = seed
        self.num_latent = num_latent
        self.num_trash = num_trash

        # Feature map for input encoding
        feature_map = ZFeatureMap(num_latent + 2 * num_trash)

        # QCNN ansatz – 3 convolution + pooling stages
        ansatz = QuantumCircuit(num_latent + 2 * num_trash)
        ansatz.compose(conv_layer(num_latent + 2 * num_trash, "c1"),
                       range(num_latent + 2 * num_trash), inplace=True)
        ansatz.compose(pool_layer(num_latent + 2 * num_trash, "p1"),
                       range(num_latent + 2 * num_trash), inplace=True)
        ansatz.compose(conv_layer(num_latent + 2 * num_trash, "c2"),
                       range(num_latent + 2 * num_trash), inplace=True)
        ansatz.compose(pool_layer(num_latent + 2 * num_trash, "p2"),
                       range(num_latent + 2 * num_trash), inplace=True)

        # Domain wall perturbation
        ansatz = domain_wall(ansatz, 0, num_latent)

        # Swap‑test ancilla for latent reconstruction
        ancilla = QuantumRegister(1, "anc")
        qc = QuantumCircuit(num_latent + 2 * num_trash + 1, 1)
        qc.add_register(ancilla)
        qc.compose(feature_map, range(num_latent + 2 * num_trash), inplace=True)
        qc.compose(ansatz, range(num_latent + 2 * num_trash), inplace=True)
        qc.h(ancilla[0])
        for i in range(num_trash):
            qc.cswap(ancilla[0], i, num_latent + i)
        qc.h(ancilla[0])
        qc.measure(ancilla[0], 0)

        # Observable for the expectation value of the ancilla
        observable = SparsePauliOp.from_list([("Z" + "I" * (num_latent + 2 * num_trash), 1)])
        estimator = Estimator()

        super().__init__(circuit=qc.decompose(),
                         observables=observable,
                         input_params=feature_map.parameters,
                         weight_params=ansatz.parameters,
                         estimator=estimator,
                         interpret=identity_interpret)

def Autoencoder() -> HybridAutoencoder:
    """Factory that returns a :class:`HybridAutoencoder` ready for training."""
    return HybridAutoencoder()

__all__ = ["HybridAutoencoder", "Autoencoder"]
