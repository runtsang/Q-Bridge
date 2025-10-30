"""
Hybrid quantum neural network mirroring the classical HybridQCNN.
Implements variational QCNN layers, a quantum autoencoder, and a sampler QNN.
"""

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap, RealAmplitudes
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator, StatevectorSampler
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.utils import algorithm_globals


class HybridQCNN:
    """
    Quantum counterpart of the classical HybridQCNN.
    Builds a QCNN ansatz, appends a quantum autoencoder sub‑circuit,
    and returns an EstimatorQNN ready for training or inference.
    """

    def __init__(self, num_qubits: int = 8, latent_dim: int = 3, num_trash: int = 2) -> None:
        algorithm_globals.random_seed = 42
        self.num_qubits = num_qubits
        self.estimator = Estimator()
        self.feature_map = ZFeatureMap(num_qubits)
        self.ansatz = self._build_ansatz()
        self.autoencoder_circ = self._build_autoencoder(latent_dim, num_trash)
        self.circuit = self._build_circuit()
        weight_params = self.ansatz.parameters + self.autoencoder_circ.parameters
        self.qnn = EstimatorQNN(
            circuit=self.circuit.decompose(),
            observables=SparsePauliOp.from_list([("Z" + "I" * (num_qubits - 1), 1)]),
            input_params=self.feature_map.parameters,
            weight_params=weight_params,
            estimator=self.estimator,
        )

    # ---------- QCNN ansatz ----------
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

        def conv_layer(num_qubits, param_prefix):
            qc = QuantumCircuit(num_qubits)
            qubits = list(range(num_qubits))
            param_index = 0
            params = ParameterVector(param_prefix, length=num_qubits * 3)
            for q1, q2 in zip(qubits[0::2], qubits[1::2]):
                qc.append(conv_circuit(params[param_index : param_index + 3]), [q1, q2])
                qc.barrier()
                param_index += 3
            for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
                qc.append(conv_circuit(params[param_index : param_index + 3]), [q1, q2])
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
            for source, sink in zip(sources, sinks):
                qc.append(pool_circuit(params[param_index : param_index + 3]), [source, sink])
                qc.barrier()
                param_index += 3
            return qc

        ansatz = QuantumCircuit(self.num_qubits)
        ansatz.compose(conv_layer(self.num_qubits, "c1"), list(range(self.num_qubits)), inplace=True)
        ansatz.compose(pool_layer([0, 1, 2, 3], [4, 5, 6, 7], "p1"), list(range(self.num_qubits)), inplace=True)
        ansatz.compose(conv_layer(self.num_qubits // 2, "c2"), list(range(self.num_qubits // 2, self.num_qubits)), inplace=True)
        ansatz.compose(pool_layer([0, 1], [2, 3], "p2"), list(range(self.num_qubits // 2, self.num_qubits)), inplace=True)
        ansatz.compose(conv_layer(self.num_qubits // 4, "c3"), list(range(self.num_qubits // 2 + self.num_qubits // 4, self.num_qubits)), inplace=True)
        ansatz.compose(pool_layer([0], [1], "p3"), list(range(self.num_qubits // 2 + self.num_qubits // 4, self.num_qubits)), inplace=True)
        return ansatz

    # ---------- Quantum autoencoder ----------
    def _build_autoencoder(self, num_latent: int, num_trash: int) -> QuantumCircuit:
        qr = QuantumRegister(num_latent + 2 * num_trash + 1, "q")
        cr = ClassicalRegister(1, "c")
        qc = QuantumCircuit(qr, cr)
        ansatz = RealAmplitudes(num_latent + num_trash, reps=5)
        qc.compose(ansatz, range(0, num_latent + num_trash), inplace=True)
        qc.barrier()
        auxiliary = num_latent + 2 * num_trash
        qc.h(auxiliary)
        for i in range(num_trash):
            qc.cswap(auxiliary, num_latent + i, num_latent + num_trash + i)
        qc.h(auxiliary)
        qc.measure(auxiliary, cr[0])
        return qc

    # ---------- Full circuit ----------
    def _build_circuit(self) -> QuantumCircuit:
        circuit = QuantumCircuit(self.num_qubits)
        circuit.compose(self.feature_map, range(self.num_qubits), inplace=True)
        circuit.compose(self.ansatz, range(self.num_qubits), inplace=True)
        circuit.compose(self.autoencoder_circ, range(self.num_qubits), inplace=True)
        return circuit

    def get_qnn(self) -> EstimatorQNN:
        """Return the constructed EstimatorQNN."""
        return self.qnn


__all__ = ["HybridQCNN"]
