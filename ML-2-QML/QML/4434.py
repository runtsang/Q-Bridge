import numpy as np
import qiskit
from qiskit import QuantumCircuit, transpile, assemble
from qiskit.providers.aer import AerSimulator
from qiskit_machine_learning.neural_networks import EstimatorQNN, SamplerQNN
from qiskit.primitives import Estimator, Sampler
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.utils import algorithm_globals

class HybridQCNNSamplerNet:
    """Quantum implementation of the hybrid QCNN + Sampler architecture."""
    def __init__(self, shots: int = 1024):
        self.backend = AerSimulator()
        self.shots = shots
        algorithm_globals.random_seed = 12345
        self._build_models()

    def _build_models(self):
        # QCNN ansatz
        self.qcnn_circuit, self.qcnn_input_params, self.qcnn_weight_params = self._build_qcnn()
        self.estimator_qnn = EstimatorQNN(
            circuit=self.qcnn_circuit,
            input_params=self.qcnn_input_params,
            weight_params=self.qcnn_weight_params,
            estimator=Estimator(backend=self.backend)
        )
        # Sampler circuit
        self.sampler_circuit, self.sampler_input_params, self.sampler_weight_params = self._build_sampler()
        self.sampler_qnn = SamplerQNN(
            circuit=self.sampler_circuit,
            input_params=self.sampler_input_params,
            weight_params=self.sampler_weight_params,
            sampler=Sampler(backend=self.backend, shots=self.shots)
        )

    def _build_qcnn(self):
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

        def conv_layer(num_qubits, param_prefix):
            qc = QuantumCircuit(num_qubits)
            qubits = list(range(num_qubits))
            param_index = 0
            params = ParameterVector(param_prefix, length=num_qubits * 3)
            for q1, q2 in zip(qubits[0::2], qubits[1::2]):
                sub = conv_circuit(params[param_index:param_index + 3])
                qc.append(sub, [q1, q2])
                qc.barrier()
                param_index += 3
            for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
                sub = conv_circuit(params[param_index:param_index + 3])
                qc.append(sub, [q1, q2])
                qc.barrier()
                param_index += 3
            return qc

        def pool_circuit(params):
            qc = QuantumCircuit(2)
            qc.rz(-np.pi / 2, 1)
            qc.cx(1, 0)
            qc.rz(params[0], 0)
            qc.ry(params[1], 1)
            qc.cx(0, 1)
            qc.ry(params[2], 1)
            return qc

        def pool_layer(sources, sinks, param_prefix):
            num_qubits = len(sources) + len(sinks)
            qc = QuantumCircuit(num_qubits)
            param_index = 0
            params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
            for src, sink in zip(sources, sinks):
                sub = pool_circuit(params[param_index:param_index + 3])
                qc.append(sub, [src, sink])
                qc.barrier()
                param_index += 3
            return qc

        # Build ansatz
        ansatz = QuantumCircuit(8)
        ansatz.compose(conv_layer(8, "c1"), list(range(8)), inplace=True)
        ansatz.compose(pool_layer([0, 1, 2, 3], [4, 5, 6, 7], "p1"), list(range(8)), inplace=True)
        ansatz.compose(conv_layer(4, "c2"), list(range(4, 8)), inplace=True)
        ansatz.compose(pool_layer([0, 1], [2, 3], "p2"), list(range(4, 8)), inplace=True)
        ansatz.compose(conv_layer(2, "c3"), list(range(6, 8)), inplace=True)
        ansatz.compose(pool_layer([0], [1], "p3"), list(range(6, 8)), inplace=True)

        # Feature map
        feature_map = ZFeatureMap(8)
        circuit = QuantumCircuit(8)
        circuit.compose(feature_map, range(8), inplace=True)
        circuit.compose(ansatz, range(8), inplace=True)

        # Observable
        observable = SparsePauliOp.from_list([("Z" + "I" * 7, 1)])
        circuit = circuit.decompose()
        return circuit, feature_map.parameters, ansatz.parameters

    def _build_sampler(self):
        inputs2 = ParameterVector("input", 2)
        weights2 = ParameterVector("weight", 4)
        qc2 = QuantumCircuit(2)
        qc2.ry(inputs2[0], 0)
        qc2.ry(inputs2[1], 1)
        qc2.cx(0, 1)
        qc2.ry(weights2[0], 0)
        qc2.ry(weights2[1], 1)
        qc2.cx(0, 1)
        qc2.ry(weights2[2], 0)
        qc2.ry(weights2[3], 1)
        return qc2, inputs2, weights2

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Run a batch of 8‑dim feature vectors through the QCNN and sampler."""
        probs_list = []
        for feat in features:
            # QCNN expectation
            qcnn_input = {p: float(v) for p, v in zip(self.qcnn_input_params, feat.tolist())}
            qcnn_weight = {p: 0.0 for p in self.qcnn_weight_params}
            expectation = self.estimator_qnn.run(qcnn_input, qcnn_weight).values[0].real

            # Sampler input: expectation as first parameter, second dummy zero
            sampler_input = {self.sampler_input_params[0]: expectation,
                             self.sampler_input_params[1]: 0.0}
            sampler_weight = {p: 0.0 for p in self.sampler_weight_params}
            sample_result = self.sampler_qnn.run(sampler_input, sampler_weight)
            samples = sample_result.samples[0]
            probs = np.bincount(samples, minlength=2) / self.shots
            probs_list.append(probs)

        return torch.tensor(probs_list, dtype=torch.float32)

__all__ = ["HybridQCNNSamplerNet"]
