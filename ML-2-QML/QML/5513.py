"""Quantum self‑attention built on a variational ansatz that combines
   convolutional, pooling and RandomLayer motifs from QCNN and Quantum‑NAT.
"""

import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer
from qiskit.circuit import ParameterVector
from qiskit.providers import Backend

class SelfAttentionGen412:
    """
    Quantum self‑attention circuit that accepts parameter vectors for rotation,
    entanglement, convolution and pooling.  The circuit is constructed on
    ``n_qubits`` and executed on a backend supplied at construction time.
    """

    def __init__(self, n_qubits: int = 4, backend: Backend | None = None):
        self.n_qubits = n_qubits
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")

    # ---------------------------------------------------------------------
    # Low‑level primitives
    # ---------------------------------------------------------------------
    def _conv_circuit(self, params: ParameterVector) -> QuantumCircuit:
        """Two‑qubit convolution block used in the QCNN example."""
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

    def _pool_circuit(self, params: ParameterVector) -> QuantumCircuit:
        """Two‑qubit pooling block used in the QCNN example."""
        qc = QuantumCircuit(2)
        qc.rz(-np.pi / 2, 1)
        qc.cx(1, 0)
        qc.rz(params[0], 0)
        qc.ry(params[1], 1)
        qc.cx(0, 1)
        qc.ry(params[2], 1)
        return qc

    def _build_circuit(self,
                       rotation_params: np.ndarray,
                       entangle_params: np.ndarray,
                       conv_params: np.ndarray,
                       pool_params: np.ndarray) -> QuantumCircuit:
        """
        Assemble the full variational circuit.

        Parameters
        ----------
        rotation_params : np.ndarray
            Shape (3*n_qubits,) – rotation angles for RX/RY/RZ on each qubit.
        entangle_params : np.ndarray
            Shape (n_qubits-1,) – angles for CRX between neighbouring qubits.
        conv_params : np.ndarray
            Shape (3 * (n_qubits//2),) – angles for each two‑qubit convolution block.
        pool_params : np.ndarray
            Shape (3 * (n_qubits//2),) – angles for each two‑qubit pooling block.
        """
        circuit = QuantumCircuit(self.qr, self.cr)

        # Rotation layer
        for i in range(self.n_qubits):
            idx = 3 * i
            circuit.rx(rotation_params[idx], i)
            circuit.ry(rotation_params[idx + 1], i)
            circuit.rz(rotation_params[idx + 2], i)

        # Entanglement layer
        for i in range(self.n_qubits - 1):
            circuit.crx(entangle_params[i], i, i + 1)

        # Convolution blocks – pairwise on (0,1), (2,3),...
        conv_pvec = ParameterVector("conv", length=len(conv_params))
        pool_pvec = ParameterVector("pool", length=len(pool_params))
        conv_idx = 0
        pool_idx = 0
        for q_pair in [(0, 1), (2, 3)]:
            conv_inst = self._conv_circuit(conv_pvec[conv_idx:conv_idx + 3]).to_instruction()
            circuit.append(conv_inst, q_pair)
            conv_idx += 3

            pool_inst = self._pool_circuit(pool_pvec[pool_idx:pool_idx + 3]).to_instruction()
            circuit.append(pool_inst, q_pair)
            pool_idx += 3

        # Measurement
        circuit.measure(self.qr, self.cr)
        return circuit

    def run(self,
            rotation_params: np.ndarray,
            entangle_params: np.ndarray,
            conv_params: np.ndarray,
            pool_params: np.ndarray,
            shots: int = 1024) -> np.ndarray:
        """
        Execute the circuit and return expectation values of Pauli‑Z on each qubit.
        """
        circuit = self._build_circuit(rotation_params,
                                      entangle_params,
                                      conv_params,
                                      pool_params)

        # Bind convolution and pooling parameters
        param_dict = {}
        for param in circuit.parameters:
            if param.name.startswith("conv"):
                index = int(param.name.split("_")[1])
                param_dict[param] = conv_params[index]
            elif param.name.startswith("pool"):
                index = int(param.name.split("_")[1])
                param_dict[param] = pool_params[index]
        circuit = circuit.assign_parameters(param_dict)

        # Execute
        job = qiskit.execute(circuit, self.backend, shots=shots)
        result = job.result()
        counts = result.get_counts(circuit)

        # Convert counts to expectation of Z
        expectations = np.zeros(self.n_qubits)
        for bitstring, cnt in counts.items():
            prob = cnt / shots
            for idx, bit in enumerate(reversed(bitstring)):
                z = 1 if bit == "0" else -1  # |0> -> +1, |1> -> -1
                expectations[idx] += z * prob
        return expectations

__all__ = ["SelfAttentionGen412"]
