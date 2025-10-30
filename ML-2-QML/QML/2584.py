"""Quantum hybrid QCNN + sampler network.

This module builds a QCNN ansatz and a separate sampler circuit,
mirroring the classical architecture above.  Both circuits are
parameterised and wrapped in Qiskit‑Machine‑Learning neural
network objects, enabling joint optimisation or separate use.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator, StatevectorSampler
from qiskit_machine_learning.neural_networks import EstimatorQNN, SamplerQNN
from qiskit_machine_learning.utils import algorithm_globals

class QCNNGen142:
    """Quantum hybrid network: QCNN ansatz + sampler head."""
    def __init__(self) -> None:
        algorithm_globals.random_seed = 12345
        self.estimator = StatevectorEstimator()
        self.sampler = StatevectorSampler()
        self.feature_map = ZFeatureMap(8)
        self._build_ansatz()
        self._build_sampler()
        self.qnn = EstimatorQNN(
            circuit=self.ansatz.decompose(),
            observables=self.observable,
            input_params=self.feature_map.parameters,
            weight_params=self.ansatz.parameters,
            estimator=self.estimator,
        )
        self.sampler_qnn = SamplerQNN(
            circuit=self.sampler_circuit,
            input_params=self.sampler_inputs,
            weight_params=self.sampler_weights,
            sampler=self.sampler,
        )
    def _build_ansatz(self) -> None:
        """Construct the QCNN ansatz with convolution and pooling layers."""
        def conv_circuit(params):
            qc = QuantumCircuit(2)
            qc.rz(-np.pi/2, 1)
            qc.cx(1, 0)
            qc.rz(params[0], 0)
            qc.ry(params[1], 1)
            qc.cx(0, 1)
            qc.ry(params[2], 1)
            qc.cx(1, 0)
            qc.rz(np.pi/2, 0)
            return qc

        def conv_layer(num_qubits, prefix):
            qc = QuantumCircuit(num_qubits)
            params = ParameterVector(prefix, length=num_qubits*3)
            for i in range(0, num_qubits, 2):
                qc.append(conv_circuit(params[i*3:(i+2)*3]), [i, i+1])
            return qc

        def pool_circuit(params):
            qc = QuantumCircuit(2)
            qc.rz(-np.pi/2, 1)
            qc.cx(1, 0)
            qc.rz(params[0], 0)
            qc.ry(params[1], 1)
            qc.cx(0, 1)
            qc.ry(params[2], 1)
            return qc

        def pool_layer(sources, sinks, prefix):
            qc = QuantumCircuit(len(sources)+len(sinks))
            params = ParameterVector(prefix, length=len(sources)*3)
            for src, snk in zip(sources, sinks):
                qc.append(pool_circuit(params[:3]), [src, snk])
                params = params[3:]
            return qc

        # Build full ansatz
        self.ansatz = QuantumCircuit(8)
        self.ansatz.compose(conv_layer(8, "c1"), inplace=True)
        self.ansatz.compose(pool_layer([0,1,2,3], [4,5,6,7], "p1"), inplace=True)
        self.ansatz.compose(conv_layer(4, "c2"), inplace=True)
        self.ansatz.compose(pool_layer([0,1], [2,3], "p2"), inplace=True)
        self.ansatz.compose(conv_layer(2, "c3"), inplace=True)
        self.ansatz.compose(pool_layer([0], [1], "p3"), inplace=True)

        # Combine with feature map
        self.ansatz.compose(self.feature_map, inplace=True)

        self.observable = SparsePauliOp.from_list([("Z" + "I"*7, 1)])
    def _build_sampler(self) -> None:
        """Construct a lightweight sampler circuit."""
        self.sampler_inputs = ParameterVector("in", 2)
        self.sampler_weights = ParameterVector("w", 4)
        qc = QuantumCircuit(2)
        qc.ry(self.sampler_inputs[0], 0)
        qc.ry(self.sampler_inputs[1], 1)
        qc.cx(0, 1)
        qc.ry(self.sampler_weights[0], 0)
        qc.ry(self.sampler_weights[1], 1)
        qc.cx(0, 1)
        qc.ry(self.sampler_weights[2], 0)
        qc.ry(self.sampler_weights[3], 1)
        self.sampler_circuit = qc
    def predict(self, features: np.ndarray) -> dict:
        """Return QCNN and sampler predictions for given classical inputs."""
        qcnn_out = self.qnn.predict(features)
        sampler_out = self.sampler_qnn.predict(features)
        return {"qcnn": qcnn_out, "sampler": sampler_out}

def QCNNGen142_factory() -> QCNNGen142:
    """Return a ready‑to‑use quantum hybrid network."""
    return QCNNGen142()

__all__ = ["QCNNGen142", "QCNNGen142_factory"]
