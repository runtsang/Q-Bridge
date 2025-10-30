from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator, StatevectorSampler as Sampler
from qiskit_machine_learning.neural_networks import EstimatorQNN as QNN, SamplerQNN as SamplerQNN
from qiskit_aer import AerSimulator

class QCNNGen333:
    """Quantum‑classical hybrid QCNN combining ansatz, sampler, and attention."""
    def __init__(self, backend: qiskit.providers.Backend | None = None) -> None:
        self.backend = backend or AerSimulator()
        # Feature map
        self.feature_map = ZFeatureMap(8)
        # Build ansatz
        self.ansatz = self._build_ansatz()
        # Observable
        self.observable = SparsePauliOp.from_list([("Z" + "I" * 7, 1)])
        # EstimatorQNN
        self.qnn = QNN(
            circuit=self.ansatz.decompose(),
            observables=self.observable,
            input_params=self.feature_map.parameters,
            weight_params=self.ansatz.parameters,
            estimator=Estimator(backend=self.backend)
        )
        # Sampler QNN
        self.sampler_qnn = self._build_sampler()
        # Attention circuit
        self.attn = self._build_attention()

    def _build_ansatz(self) -> QuantumCircuit:
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

        def conv_layer(num_qubits, name):
            qc = QuantumCircuit(num_qubits, name=name)
            params = ParameterVector(name, length=num_qubits * 3)
            for i in range(0, num_qubits, 2):
                qc.compose(conv_circuit(params[i:i+3]), [i, i+1], inplace=True)
                qc.barrier()
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

        def pool_layer(num_qubits, name):
            qc = QuantumCircuit(num_qubits, name=name)
            params = ParameterVector(name, length=num_qubits // 2 * 3)
            for i in range(0, num_qubits, 2):
                qc.compose(pool_circuit(params[(i//2)*3:(i//2+1)*3]), [i, i+1], inplace=True)
                qc.barrier()
            return qc

        ansatz = QuantumCircuit(8)
        ansatz.compose(conv_layer(8, "c1"), inplace=True)
        ansatz.compose(pool_layer(8, "p1"), inplace=True)
        ansatz.compose(conv_layer(4, "c2"), inplace=True)
        ansatz.compose(pool_layer(4, "p2"), inplace=True)
        ansatz.compose(conv_layer(2, "c3"), inplace=True)
        ansatz.compose(pool_layer(2, "p3"), inplace=True)
        return ansatz

    def _build_sampler(self) -> SamplerQNN:
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
        sampler = Sampler(backend=self.backend)
        return SamplerQNN(circuit=qc, input_params=inputs, weight_params=weights, sampler=sampler)

    def _build_attention(self):
        class QuantumSelfAttention:
            def __init__(self, n_qubits: int):
                self.n_qubits = n_qubits
                self.qr = QuantumRegister(n_qubits, "q")
                self.cr = ClassicalRegister(n_qubits, "c")

            def _build_circuit(self, rotation_params: np.ndarray, entangle_params: np.ndarray):
                qc = QuantumCircuit(self.qr, self.cr)
                for i in range(self.n_qubits):
                    qc.rx(rotation_params[3 * i], i)
                    qc.ry(rotation_params[3 * i + 1], i)
                    qc.rz(rotation_params[3 * i + 2], i)
                for i in range(self.n_qubits - 1):
                    qc.crx(entangle_params[i], i, i + 1)
                qc.measure(self.qr, self.cr)
                return qc

            def run(self, rotation_params: np.ndarray, entangle_params: np.ndarray, shots: int = 1024):
                qc = self._build_circuit(rotation_params, entangle_params)
                job = execute(qc, self.backend, shots=shots)
                return job.result().get_counts(qc)

        return QuantumSelfAttention(n_qubits=4)

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """Return classification probabilities from the quantum ansatz."""
        return self.qnn.predict(inputs)

    def sample(self, inputs: np.ndarray) -> dict:
        """Return sampler counts for the first two features of each input."""
        return self.sampler_qnn.predict(inputs)

    def attention(self, inputs: np.ndarray) -> dict:
        """Run the attention circuit with random parameters."""
        rot = np.random.rand(12)
        ent = np.random.rand(3)
        return self.attn.run(rot, ent)

    def run(self, inputs: np.ndarray) -> dict:
        """Convenience wrapper returning all components."""
        return {
            "prediction": self.predict(inputs),
            "sampler_counts": self.sample(inputs),
            "attention_counts": self.attention(inputs)
        }

def QCNN() -> QCNNGen333:
    """Factory returning a configured QCNNGen333 instance."""
    return QCNNGen333()

__all__ = ["QCNN", "QCNNGen333"]
