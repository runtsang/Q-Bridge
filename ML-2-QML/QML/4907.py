"""Quantum‑inspired self‑attention with QCNN‑style convolution and pooling.

The circuit implements a parameterised Qiskit circuit that mirrors the classical
SelfAttention helper but enriches it with hierarchical layers taken from the QCNN
example.  It can be used directly or wrapped in a SamplerQNN/EstimatorQNN
object for hybrid training.
"""

import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import ParameterVector
from qiskit_machine_learning.neural_networks import SamplerQNN, EstimatorQNN
from qiskit.primitives import StatevectorSampler, StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp


class _HybridQuantumSelfAttention:
    """Quantum circuit that produces a probability distribution over ``embed_dim`` outcomes."""

    def __init__(self, n_qubits: int = 8) -> None:
        self.n_qubits = n_qubits
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")

    # ------------------------------------------------------------------
    # QCNN‑style primitives
    # ------------------------------------------------------------------
    def _conv_circuit(self, params):
        """Convolution step used in QCNN."""
        target = QuantumCircuit(2)
        target.rz(-np.pi / 2, 1)
        target.cx(1, 0)
        target.rz(params[0], 0)
        target.ry(params[1], 1)
        target.cx(0, 1)
        target.ry(params[2], 1)
        target.cx(1, 0)
        target.rz(np.pi / 2, 0)
        return target

    def _pool_circuit(self, params):
        """Pooling step used in QCNN."""
        target = QuantumCircuit(2)
        target.rz(-np.pi / 2, 1)
        target.cx(1, 0)
        target.rz(params[0], 0)
        target.ry(params[1], 1)
        target.cx(0, 1)
        target.ry(params[2], 1)
        return target

    def _conv_layer(self, num_qubits, param_prefix):
        """Build a convolutional layer over all qubits."""
        qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
        qubits = list(range(num_qubits))
        param_index = 0
        params = ParameterVector(param_prefix, length=num_qubits * 3)
        for q1, q2 in zip(qubits[0::2], qubits[1::2]):
            qc = qc.compose(
                self._conv_circuit(params[param_index : param_index + 3]), [q1, q2]
            )
            qc.barrier()
            param_index += 3
        for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
            qc = qc.compose(
                self._conv_circuit(params[param_index : param_index + 3]), [q1, q2]
            )
            qc.barrier()
            param_index += 3
        qc_inst = qc.to_instruction()
        qc = QuantumCircuit(num_qubits)
        qc.append(qc_inst, qubits)
        return qc

    def _pool_layer(self, sources, sinks, param_prefix):
        """Build a pooling layer between two lists of qubits."""
        num_qubits = len(sources) + len(sinks)
        qc = QuantumCircuit(num_qubits, name="Pooling Layer")
        param_index = 0
        params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
        for source, sink in zip(sources, sinks):
            qc = qc.compose(
                self._pool_circuit(params[param_index : param_index + 3]), [source, sink]
            )
            qc.barrier()
            param_index += 3
        qc_inst = qc.to_instruction()
        qc = QuantumCircuit(num_qubits)
        qc.append(qc_inst, range(num_qubits))
        return qc

    # ------------------------------------------------------------------
    # Circuit construction
    # ------------------------------------------------------------------
    def _build_circuit(self, rotation_params: np.ndarray, entangle_params: np.ndarray) -> QuantumCircuit:
        """Assemble the full self‑attention circuit."""
        circuit = QuantumCircuit(self.qr, self.cr)

        # Apply per‑qubit rotations (acts as query/key/value preparation)
        for i in range(self.n_qubits):
            circuit.rx(rotation_params[3 * i], i)
            circuit.ry(rotation_params[3 * i + 1], i)
            circuit.rz(rotation_params[3 * i + 2], i)

        # Entangling gates (CRX) to correlate qubits
        for i in range(self.n_qubits - 1):
            circuit.crx(entangle_params[i], i, i + 1)

        # QCNN‑style convolution & pooling
        conv_layer = self._conv_layer(self.n_qubits, "c")
        circuit.append(conv_layer, list(range(self.n_qubits)))

        # Simple pooling between the first and second halves
        pool_layer = self._pool_layer([0, 1, 2, 3], [4, 5, 6, 7], "p")
        circuit.append(pool_layer, list(range(self.n_qubits)))

        circuit.measure(self.qr, self.cr)
        return circuit

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------
    def run(
        self,
        backend,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        shots: int = 1024,
    ) -> Dict[str, int]:
        """
        Execute the self‑attention circuit on a given backend.

        Parameters
        ----------
        backend : qiskit.providers.Backend
            The quantum backend to use (e.g. Aer simulator or real device).
        rotation_params : np.ndarray
            Parameters for the rotation gates.
        entangle_params : np.ndarray
            Parameters for the CRX entangling gates.
        shots : int, default=1024
            Number of measurement shots.

        Returns
        -------
        dict
            Counts dictionary mapping bit‑strings to frequencies.
        """
        circuit = self._build_circuit(rotation_params, entangle_params)
        job = qiskit.execute(circuit, backend, shots=shots)
        result = job.result()
        return result.get_counts(circuit)


def HybridQuantumSelfAttention(n_qubits: int = 8) -> _HybridQuantumSelfAttention:
    """Factory returning a configured :class:`_HybridQuantumSelfAttention` instance."""
    return _HybridQuantumSelfAttention(n_qubits)


__all__ = ["HybridQuantumSelfAttention"]
