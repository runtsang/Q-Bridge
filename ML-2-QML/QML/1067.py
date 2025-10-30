import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN

class QCNNHybrid(EstimatorQNN):
    """Variational QCNN with multi‑observable readout and custom feature map."""
    def __init__(self):
        # Feature map
        feature_map = ZFeatureMap(num_qubits=8, reps=2, entanglement='circular')
        # Ansatz: hierarchical convolution‑pooling layers
        ansatz = self._build_ansatz()
        # Observables: Z on each qubit for vector readout
        observables = SparsePauliOp.from_list([
            ("Z" + "I" * 7, 1),
            ("I" + "Z" + "I" * 6, 1),
            ("I" * 2 + "Z" + "I" * 5, 1)
        ])
        estimator = Estimator()
        super().__init__(circuit=ansatz.decompose(),
                         observables=observables,
                         input_params=feature_map.parameters,
                         weight_params=ansatz.parameters,
                         estimator=estimator)

    def _build_ansatz(self) -> QuantumCircuit:
        """Construct a hierarchical QCNN ansatz with convolution and pooling."""
        qc = QuantumCircuit(8, name="QCNN_Ansatz")

        def conv_layer(qubits: list[int], prefix: str):
            layer = QuantumCircuit(8, name=f"Conv_{prefix}")
            params = ParameterVector(prefix, len(qubits) * 3)
            for i in range(0, len(qubits), 2):
                q1, q2 = qubits[i], qubits[i+1]
                layer.compose(self._conv_circuit(params[i*3:(i+1)*3]), [q1, q2], inplace=True)
                layer.barrier()
            return layer

        def pool_layer(qubits: list[int], prefix: str):
            layer = QuantumCircuit(8, name=f"Pool_{prefix}")
            params = ParameterVector(prefix, len(qubits) * 3)
            for i in range(0, len(qubits), 2):
                q1, q2 = qubits[i], qubits[i+1]
                layer.compose(self._pool_circuit(params[i*3:(i+1)*3]), [q1, q2], inplace=True)
                layer.barrier()
            return layer

        # First convolution over all qubits
        qc.compose(conv_layer(list(range(8)), "c1"), inplace=True)
        # First pooling (reduce to 4 qubits)
        qc.compose(pool_layer([0,1,2,3], "p1"), inplace=True)
        qc.compose(pool_layer([4,5,6,7], "p2"), inplace=True)
        # Second convolution over remaining 4 qubits
        qc.compose(conv_layer([0,1,2,3], "c2"), inplace=True)
        # Second pooling (reduce to 2 qubits)
        qc.compose(pool_layer([0,1], "p3"), inplace=True)
        # Third convolution over 2 qubits
        qc.compose(conv_layer([0,1], "c3"), inplace=True)
        # Third pooling (measure single qubit)
        qc.compose(pool_layer([0,1], "p4"), inplace=True)

        return qc

    def _conv_circuit(self, params: ParameterVector) -> QuantumCircuit:
        """Single 2‑qubit convolution unit."""
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

    def _pool_circuit(self, params: ParameterVector) -> QuantumCircuit:
        """Single 2‑qubit pooling unit."""
        qc = QuantumCircuit(2)
        qc.rz(-np.pi/2,1)
        qc.cx(1,0)
        qc.rz(params[0],0)
        qc.ry(params[1],1)
        qc.cx(0,1)
        qc.ry(params[2],1)
        return qc

def QCNNHybridFactory() -> QCNNHybrid:
    """Convenience factory that returns a ready‑to‑train QCNNHybrid."""
    return QCNNHybrid()

__all__ = ["QCNNHybrid", "QCNNHybridFactory"]
