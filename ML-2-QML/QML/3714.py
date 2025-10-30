from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.providers import Backend
import math

class HybridQuantumAttentionQCNN:
    """Hybrid QCNN + attention variational circuit."""
    def __init__(self, n_qubits: int = 4) -> None:
        self.n_qubits = n_qubits
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")
        self.feature_map = ZFeatureMap(n_qubits)
        self.cnn_circuit = self._build_qcnn()
        self.attention_circuit = self._build_attention_layer()
        self.circuit = self._build_circuit()

    def _build_conv_block(self, params):
        target = QuantumCircuit(2)
        target.rz(-math.pi / 2, 1)
        target.cx(1, 0)
        target.rz(params[0], 0)
        target.ry(params[1], 1)
        target.cx(0, 1)
        target.ry(params[2], 1)
        target.cx(1, 0)
        target.rz(math.pi / 2, 0)
        return target

    def _build_pool_block(self, params):
        target = QuantumCircuit(2)
        target.rz(-math.pi / 2, 1)
        target.cx(1, 0)
        target.rz(params[0], 0)
        target.ry(params[1], 1)
        target.cx(0, 1)
        target.ry(params[2], 1)
        return target

    def _build_qcnn(self):
        qc = QuantumCircuit(self.n_qubits)
        # single convolution layer (two‑qubit blocks)
        conv_params = ParameterVector("c", length=self.n_qubits * 3)
        for i in range(0, self.n_qubits, 2):
            block = self._build_conv_block(conv_params[i * 3:(i + 2) * 3])
            qc.append(block, [i, i + 1])
        # single pooling layer
        pool_params = ParameterVector("p", length=(self.n_qubits // 2) * 3)
        for i in range(0, self.n_qubits, 2):
            block = self._build_pool_block(pool_params[i // 2 * 3:(i // 2 + 1) * 3])
            qc.append(block, [i, i + 1])
        return qc

    def _build_attention_layer(self):
        """Attention‑like layer – controlled rotations between neighbours."""
        qc = QuantumCircuit(self.n_qubits)
        params = ParameterVector("att", length=self.n_qubits - 1)
        for i in range(self.n_qubits - 1):
            qc.crx(params[i], i, i + 1)
        return qc

    def _build_circuit(self):
        qc = QuantumCircuit(self.qr, self.cr)
        qc.compose(self.feature_map, self.qr, inplace=True)
        qc.compose(self.cnn_circuit, self.qr, inplace=True)
        qc.compose(self.attention_circuit, self.qr, inplace=True)
        qc.measure(self.qr, self.cr)
        return qc

    def run(self,
            backend: Backend = None,
            attention_params: dict = None,
            conv_params: dict = None,
            pool_params: dict = None,
            shots: int = 1024):
        """
        Execute the circuit on the supplied backend.

        Parameters
        ----------
        backend : qiskit.providers.Backend, optional
            Any Qiskit backend (simulator or real device).  If None,
            a local QASM simulator is used.
        attention_params : dict
            Mapping from attention parameter names to values.
        conv_params, pool_params : dict, optional
            Parameter mappings for the convolution and pooling stages.
        shots : int, default 1024
            Number of measurement shots.

        Returns
        -------
        dict
            Measurement counts for each computational basis string.
        """
        if backend is None:
            backend = Aer.get_backend("qasm_simulator")
        bound = self.circuit.bind_parameters(attention_params or {})
        if conv_params:
            bound = bound.bind_parameters(conv_params)
        if pool_params:
            bound = bound.bind_parameters(pool_params)
        job = execute(bound, backend, shots=shots)
        return job.result().get_counts(bound)

__all__ = ["HybridQuantumAttentionQCNN"]
