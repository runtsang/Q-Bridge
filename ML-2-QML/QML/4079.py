import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator as Estimator, StatevectorSampler as Sampler
from qiskit_machine_learning.neural_networks import EstimatorQNN

class HybridSamplerQNNQuantum:
    """
    Quantum neural network mirroring the classical HybridSamplerQNN.
    The circuit consists of an 8‑qubit Z‑feature map followed by a
    convolution‑pool‑convolution hierarchy that emulates the QCNN
    architecture.  A `StatevectorSampler` is used to obtain a
    probability distribution that can be fed into a classical
    loss function.
    """
    def __init__(self):
        # 8‑qubit feature map
        self.feature_map = ZFeatureMap(8, reps=1, insert_barriers=False)

        # Build the hierarchical ansatz
        self.ansatz = self._build_ansatz()

        # Observable for the sampler (Z on qubit 0)
        self.observable = SparsePauliOp.from_list([("Z" + "I" * 7, 1)])

        # Estimator and sampler primitives
        self.estimator = Estimator()
        self.sampler = Sampler()

        # Quantum neural network
        self.qnn = EstimatorQNN(
            circuit=self.ansatz,
            observables=self.observable,
            input_params=self.feature_map.parameters,
            weight_params=self.ansatz.parameters,
            estimator=self.estimator,
        )

    # ------------------------------------------------------------------
    # Convolution and pooling subcircuits
    # ------------------------------------------------------------------
    def _conv_circuit(self, params, qubits):
        qc = QuantumCircuit(len(qubits))
        qc.rz(-np.pi / 2, qubits[1])
        qc.cx(qubits[1], qubits[0])
        qc.rz(params[0], qubits[0])
        qc.ry(params[1], qubits[1])
        qc.cx(qubits[0], qubits[1])
        qc.ry(params[2], qubits[1])
        qc.cx(qubits[1], qubits[0])
        qc.rz(np.pi / 2, qubits[0])
        return qc

    def _pool_circuit(self, params, qubits):
        qc = QuantumCircuit(len(qubits))
        qc.rz(-np.pi / 2, qubits[1])
        qc.cx(qubits[1], qubits[0])
        qc.rz(params[0], qubits[0])
        qc.ry(params[1], qubits[1])
        qc.cx(qubits[0], qubits[1])
        qc.ry(params[2], qubits[1])
        return qc

    def _conv_layer(self, num_qubits, prefix):
        qc = QuantumCircuit(num_qubits)
        params = ParameterVector(prefix, length=num_qubits * 3)
        idx = 0
        for q1, q2 in zip(range(0, num_qubits, 2), range(1, num_qubits, 2)):
            sub = self._conv_circuit(params[idx:idx+3], [q1, q2])
            qc.append(sub.to_instruction(), [q1, q2])
            qc.barrier()
            idx += 3
        return qc

    def _pool_layer(self, sources, sinks, prefix):
        num_qubits = len(sources) + len(sinks)
        qc = QuantumCircuit(num_qubits)
        params = ParameterVector(prefix, length=num_qubits // 2 * 3)
        idx = 0
        for src, snk in zip(sources, sinks):
            sub = self._pool_circuit(params[idx:idx+3], [src, snk])
            qc.append(sub.to_instruction(), [src, snk])
            qc.barrier()
            idx += 3
        return qc

    # ------------------------------------------------------------------
    # Assemble the full ansatz
    # ------------------------------------------------------------------
    def _build_ansatz(self):
        qc = QuantumCircuit(8)
        qc.append(self._conv_layer(8, "c1"), range(8))
        qc.append(self._pool_layer([0,1,2,3], [4,5,6,7], "p1"), range(8))
        qc.append(self._conv_layer(4, "c2"), [4,5,6,7])
        qc.append(self._pool_layer([0,1], [2,3], "p2"), [4,5,6,7])
        qc.append(self._conv_layer(2, "c3"), [6,7])
        qc.append(self._pool_layer([0], [1], "p3"), [6,7])
        return qc

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def evaluate(self, input_data: np.ndarray) -> np.ndarray:
        """
        Compute sampler probabilities for a batch of feature vectors.

        Parameters
        ----------
        input_data : np.ndarray
            Array of shape (batch, 8) containing values for the 8
            feature‑map parameters.

        Returns
        -------
        np.ndarray
            Probability distribution of shape (batch, 2) obtained from the
            state‑vector sampler.
        """
        return self.sampler.sample(self.qnn, input_data).data

__all__ = ["HybridSamplerQNNQuantum"]
