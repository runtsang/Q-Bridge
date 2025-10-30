import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.primitives import Estimator, StatevectorSampler
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.utils import algorithm_globals
import strawberryfields as sf
from strawberryfields.ops import BSgate, Dgate, Kgate, Rgate, Sgate

# --------------------------------------------------------------------------- #
# QCNN‑style quantum circuit components
# --------------------------------------------------------------------------- #

def _conv_circuit(params):
    qc = QuantumCircuit(2)
    qc.rz(-np.pi/2, 1)
    qc.cx(1,0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0,1)
    qc.ry(params[2], 1)
    qc.cx(1,0)
    qc.rz(np.pi/2, 0)
    return qc

def _pool_circuit(params):
    qc = QuantumCircuit(2)
    qc.rz(-np.pi/2, 1)
    qc.cx(1,0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0,1)
    qc.ry(params[2], 1)
    return qc

def conv_layer(num_qubits, prefix):
    qc = QuantumCircuit(num_qubits)
    params = ParameterVector(prefix, length=num_qubits//2*3)
    idx = 0
    for q1, q2 in zip(range(0, num_qubits, 2), range(1, num_qubits, 2)):
        sub = _conv_circuit(params[idx:idx+3])
        qc.append(sub, [q1, q2])
        idx += 3
    return qc

def pool_layer(sources, sinks, prefix):
    num_qubits = len(sources)+len(sinks)
    params = ParameterVector(prefix, length=len(sources)*3)
    qc = QuantumCircuit(num_qubits)
    idx = 0
    for src, snk in zip(sources, sinks):
        sub = _pool_circuit(params[idx:idx+3])
        qc.append(sub, [src, snk])
        idx += 3
    return qc

# --------------------------------------------------------------------------- #
# Photonic fraud‑detection style circuit (Strawberry Fields)
# --------------------------------------------------------------------------- #

def build_photonic_fraud_circuit(params):
    """Return a Strawberry Fields program implementing the photonic fraud
    detection layer described in the original reference."""
    prog = sf.Program(2)
    with prog.context as q:
        BSgate(params.bs_theta, params.bs_phi) | (q[0], q[1])
        for i, phase in enumerate(params.phases):
            Rgate(phase) | q[i]
        for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
            Sgate(max(-5, min(5, r)), phi) | q[i]
        BSgate(params.bs_theta, params.bs_phi) | (q[0], q[1])
        for i, phase in enumerate(params.phases):
            Rgate(phase) | q[i]
        for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
            Dgate(max(-5, min(5, r)), phi) | q[i]
        for i, k in enumerate(params.kerr):
            Kgate(max(-1, min(1, k))) | q[i]
    return prog

# --------------------------------------------------------------------------- #
# Hybrid QCNN‑style quantum circuit construction
# --------------------------------------------------------------------------- #

def build_hybrid_qnn(num_qubits=8):
    """
    Constructs a hybrid quantum neural network that combines
    QCNN convolution‑pooling layers with a photonic fraud‑detection ansatz.
    Returns an EstimatorQNN ready for training with a classical optimiser.
    """
    algorithm_globals.random_seed = 12345
    estimator = Estimator()

    # Feature map
    feature_map = ZFeatureMap(num_qubits)

    # Ansatz built from QCNN layers
    ansatz = QuantumCircuit(num_qubits)
    ansatz.compose(conv_layer(num_qubits, "c1"), range(num_qubits), inplace=True)
    ansatz.compose(pool_layer(list(range(num_qubits//2)), list(range(num_qubits//2, num_qubits)), "p1"),
                   range(num_qubits), inplace=True)

    # Photonic fraud‑detection block (placeholder)
    # In a full implementation this would be a sub‑instruction from
    # build_photonic_fraud_circuit. Here we simply add an extra convolution
    # for demonstration purposes.
    ansatz.compose(conv_layer(num_qubits//2, "c2"), range(num_qubits//2), inplace=True)
    ansatz.compose(pool_layer(list(range(num_qubits//4)), list(range(num_qubits//4, num_qubits//2)), "p2"),
                   range(num_qubits//2), inplace=True)

    # Combine feature map and ansatz
    circuit = QuantumCircuit(num_qubits)
    circuit.compose(feature_map, range(num_qubits), inplace=True)
    circuit.compose(ansatz, range(num_qubits), inplace=True)

    # Observable: single‑qubit Z on the first qubit
    observable = SparsePauliOp.from_list([("Z" + "I" * (num_qubits-1), 1)])

    qnn = EstimatorQNN(
        circuit=circuit.decompose(),
        observables=observable,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        estimator=estimator,
    )
    return qnn

# --------------------------------------------------------------------------- #
# Fast quantum estimator wrapper
# --------------------------------------------------------------------------- #

class FastQuantumEstimator:
    """
    Thin wrapper around qiskit.primitives.Estimator that mimics the
    FastBaseEstimator interface from the classical reference.
    """
    def __init__(self, circuit: QuantumCircuit):
        self._circuit = circuit
        self._params = list(circuit.parameters)
        self._estimator = Estimator()

    def _bind(self, values):
        if len(values)!= len(self._params):
            raise ValueError("Parameter count mismatch.")
        mapping = dict(zip(self._params, values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(self, observables, parameter_sets):
        results = []
        for vals in parameter_sets:
            state = Statevector.from_instruction(self._bind(vals))
            results.append([state.expectation_value(obs) for obs in observables])
        return results

__all__ = [
    "build_hybrid_qnn",
    "FastQuantumEstimator",
    "build_photonic_fraud_circuit",
]
