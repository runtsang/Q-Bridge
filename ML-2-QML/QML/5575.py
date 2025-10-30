import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator as Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN

class HybridQCNN(EstimatorQNN):
    """Variational QCNN circuit with 4 qubits, inspired by the QCNN example.
    The circuit consists of a feature map followed by a single convolutional‑pooling
    block and a 4‑qubit parameterized ansatz.  It is compatible with the
    classical projection of size 4 produced by :class:`QCNN__gen482_ml.HybridQCNN`.
    """
    def __init__(self) -> None:
        # Convolution gate for two qubits
        def conv_circuit(params):
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

        # Pooling gate for two qubits
        def pool_circuit(params):
            qc = QuantumCircuit(2)
            qc.rz(-np.pi/2,1)
            qc.cx(1,0)
            qc.rz(params[0],0)
            qc.ry(params[1],1)
            qc.cx(0,1)
            qc.ry(params[2],1)
            return qc

        # Parameter vectors
        conv_params1 = ParameterVector("c1_0", 3)
        pool_params1 = ParameterVector("p1_0", 3)
        conv_params2 = ParameterVector("c1_1", 3)
        pool_params2 = ParameterVector("p1_1", 3)

        # Build ansatz: two conv–pool blocks on 4 qubits
        ansatz = QuantumCircuit(4)
        ansatz.compose(conv_circuit(conv_params1), [0,1], inplace=True)
        ansatz.compose(pool_circuit(pool_params1), [0,1], inplace=True)
        ansatz.compose(conv_circuit(conv_params2), [2,3], inplace=True)
        ansatz.compose(pool_circuit(pool_params2), [2,3], inplace=True)

        # Feature map with 4 qubits
        feature_map = ZFeatureMap(4)

        # Full circuit
        circuit = QuantumCircuit(4)
        circuit.compose(feature_map, inplace=True)
        circuit.compose(ansatz, inplace=True)

        # Observable: ZZZZ
        observable = SparsePauliOp.from_list([("Z"*4, 1)])

        estimator = Estimator()
        super().__init__(
            circuit=circuit.decompose(),
            observables=observable,
            input_params=feature_map.parameters,
            weight_params=ansatz.parameters,
            estimator=estimator,
        )

__all__ = ["HybridQCNN"]
