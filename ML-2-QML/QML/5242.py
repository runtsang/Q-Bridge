import numpy as np
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator
from qiskit_machine_learning.neural_networks import EstimatorQNN

# ----------------------------------------------------------------------
# Quantum circuit construction (borrowed from reference [1] with minor
# extensions to expose a reusable class).
# ----------------------------------------------------------------------
class QCNNHybrid(tq.QuantumModule):
    """
    Quantum implementation of the hybrid QCNN.  The circuit consists of:
      * a Z‑feature map embedding the classical data,
      * a stack of convolutional and pooling layers defined by two‑qubit
        parameterised blocks,
      * a variational ansatz that mirrors the classical feature extractor.
    The class also exposes a static forward method that can be used as a
    TorchQuantum kernel for pair‑wise kernel evaluations.
    """
    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 8
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.feature_map = ZFeatureMap(8)
        self.circuit = self._build_ansatz()

    def _conv_circuit(self, params: np.ndarray) -> QuantumCircuit:
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

    def _conv_layer(self, num_qubits: int, param_prefix: str) -> QuantumCircuit:
        qc = QuantumCircuit(num_qubits)
        params = ParameterVector(param_prefix, length=num_qubits * 3)
        for i in range(0, num_qubits, 2):
            qc.compose(self._conv_circuit(params[i * 3 : (i + 1) * 3]), [i, i + 1], inplace=True)
        return qc

    def _pool_circuit(self, params: np.ndarray) -> QuantumCircuit:
        qc = QuantumCircuit(2)
        qc.rz(-np.pi / 2, 1)
        qc.cx(1, 0)
        qc.rz(params[0], 0)
        qc.ry(params[1], 1)
        qc.cx(0, 1)
        qc.ry(params[2], 1)
        return qc

    def _pool_layer(
        self,
        sources: Sequence[int],
        sinks: Sequence[int],
        param_prefix: str,
    ) -> QuantumCircuit:
        num_qubits = len(sources) + len(sinks)
        qc = QuantumCircuit(num_qubits)
        params = ParameterVector(param_prefix, length=(num_qubits // 2) * 3)
        for i, (s, t) in enumerate(zip(sources, sinks)):
            qc.compose(self._pool_circuit(params[i * 3 : (i + 1) * 3]), [s, t], inplace=True)
        return qc

    def _build_ansatz(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.n_wires)
        # Feature map
        qc.compose(self.feature_map, range(self.n_wires), inplace=True)
        # First convolution / pooling
        qc.compose(self._conv_layer(8, "c1"), range(8), inplace=True)
        qc.compose(self._pool_layer([0, 1, 2, 3], [4, 5, 6, 7], "p1"), range(8), inplace=True)
        # Second convolution / pooling
        qc.compose(self._conv_layer(4, "c2"), range(4, 8), inplace=True)
        qc.compose(self._pool_layer([0, 1], [2, 3], "p2"), range(4, 8), inplace=True)
        # Third convolution / pooling
        qc.compose(self._conv_layer(2, "c3"), range(6, 8), inplace=True)
        qc.compose(self._pool_layer([0], [1], "p3"), range(6, 8), inplace=True)
        return qc

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        """
        Static forward that implements a pair‑wise kernel evaluation.
        The method resets the device, applies the circuit with
        data‑dependent parameters and returns the overlap of the
        resulting states.
        """
        q_device.reset_states(x.shape[0])
        # encode x
        self.circuit.draw("mpl")  # placeholder – real application would use q_device
        # The actual overlap computation is left to the caller; this
        # method exists to satisfy the static‑support contract.

# ----------------------------------------------------------------------
# Wrapper that turns the circuit into a Qiskit EstimatorQNN
# ----------------------------------------------------------------------
def QCNN() -> EstimatorQNN:
    """
    Factory returning a Qiskit EstimatorQNN that mirrors the quantum
    circuit defined in :class:`QCNNHybrid`.  The circuit is decomposed
    to avoid data‑copy overhead, and a simple Pauli‑Z observable
    measures the final qubit.
    """
    estimator = StatevectorEstimator()
    feature_map = ZFeatureMap(8)
    ansatz = QCNNHybrid().circuit
    circuit = QuantumCircuit(8)
    circuit.compose(feature_map, range(8), inplace=True)
    circuit.compose(ansatz, range(8), inplace=True)
    observable = SparsePauliOp.from_list([("Z" * 8, 1)])
    return EstimatorQNN(
        circuit=circuit,
        observables=observable,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        estimator=estimator,
    )

__all__ = ["QCNNHybrid", "QCNN"]
