from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap, RealAmplitudes
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN

class CombinedEstimatorQNN:
    """Quantum neural network that mirrors the classical hybrid architecture:
    a feature map, a QCNN ansatz, an auto‑encoder style subcircuit, and a self‑attention block."""
    def __init__(self):
        # Backend estimator
        self.estimator = StatevectorEstimator()
        # Feature map
        self.feature_map = ZFeatureMap(8)
        # QCNN layers
        conv_layer = self._conv_layer(8, "c1")
        pool_layer = self._pool_layer([0,1,2,3], [4,5,6,7], "p1")
        # Auto‑encoder subcircuit
        auto_circ = self._autoencoder_circuit(3, 2)
        # Self‑attention subcircuit
        sa_circ = self._self_attention_circuit(4)
        # Assemble ansatz
        ansatz = QuantumCircuit(8)
        ansatz.compose(conv_layer, range(8), inplace=True)
        ansatz.compose(pool_layer, range(8), inplace=True)
        ansatz.compose(auto_circ, range(8), inplace=True)
        ansatz.compose(sa_circ, range(8), inplace=True)
        # Full circuit
        circuit = QuantumCircuit(8)
        circuit.compose(self.feature_map, range(8), inplace=True)
        circuit.compose(ansatz, range(8), inplace=True)
        # Observable
        observable = SparsePauliOp.from_list([("Z"*8, 1)])
        # Parameters
        input_params = self.feature_map.parameters
        weight_params = ansatz.parameters
        # QNN wrapper
        self.qnn = EstimatorQNN(
            circuit=circuit.decompose(),
            observables=observable,
            input_params=input_params,
            weight_params=weight_params,
            estimator=self.estimator,
        )

    def _conv_layer(self, num_qubits: int, prefix: str) -> QuantumCircuit:
        """Convolutional layer composed of 2‑qubit blocks."""
        qc = QuantumCircuit(num_qubits)
        param_index = 0
        params = ParameterVector(f"{prefix}_p", length=num_qubits * 3)
        for q1, q2 in zip(range(0, num_qubits, 2), range(1, num_qubits, 2)):
            block = self._conv_block(q1, q2, params[param_index:param_index+3])
            qc.append(block, [q1, q2])
            param_index += 3
        return qc

    def _conv_block(self, q1: int, q2: int, params) -> QuantumCircuit:
        """3‑parameter 2‑qubit convolution block."""
        block = QuantumCircuit(2)
        block.rz(-np.pi/2, q2)
        block.cx(q2, q1)
        block.rz(params[0], q1)
        block.ry(params[1], q2)
        block.cx(q1, q2)
        block.ry(params[2], q2)
        block.cx(q2, q1)
        block.rz(np.pi/2, q1)
        return block

    def _pool_layer(self, sources, sinks, prefix: str) -> QuantumCircuit:
        """Pooling layer that entangles source and sink qubits."""
        num_qubits = len(sources) + len(sinks)
        qc = QuantumCircuit(num_qubits)
        param_index = 0
        params = ParameterVector(f"{prefix}_p", length=len(sources) * 3)
        for src, snk in zip(sources, sinks):
            block = self._pool_block(src, snk, params[param_index:param_index+3])
            qc.append(block, [src, snk])
            param_index += 3
        return qc

    def _pool_block(self, q1: int, q2: int, params) -> QuantumCircuit:
        """3‑parameter 2‑qubit pooling block."""
        block = QuantumCircuit(2)
        block.rz(-np.pi/2, q2)
        block.cx(q2, q1)
        block.rz(params[0], q1)
        block.ry(params[1], q2)
        block.cx(q1, q2)
        block.ry(params[2], q2)
        return block

    def _autoencoder_circuit(self, num_latent: int, num_trash: int) -> QuantumCircuit:
        """Auto‑encoder style subcircuit using RealAmplitudes and a swap test."""
        total_qubits = num_latent + 2 * num_trash + 1
        qc = QuantumCircuit(total_qubits)
        # Ansatz
        ansatz = RealAmplitudes(num_latent + num_trash, reps=5)
        qc.append(ansatz, range(0, num_latent + num_trash))
        qc.barrier()
        aux = num_latent + 2 * num_trash
        # Swap test
        qc.h(aux)
        for i in range(num_trash):
            qc.cswap(aux, num_latent + i, num_latent + num_trash + i)
        qc.h(aux)
        qc.measure(aux, 0)  # dummy measurement to keep circuit tidy
        return qc

    def _self_attention_circuit(self, n_qubits: int) -> QuantumCircuit:
        """Self‑attention style block built from single‑qubit rotations and controlled‑rotations."""
        qr = QuantumRegister(n_qubits)
        cr = ClassicalRegister(n_qubits)
        qc = QuantumCircuit(qr, cr)
        rot_params = ParameterVector("sa_rot", length=3*n_qubits)
        ent_params = ParameterVector("sa_ent", length=n_qubits-1)
        for i in range(n_qubits):
            qc.rx(rot_params[3*i], i)
            qc.ry(rot_params[3*i+1], i)
            qc.rz(rot_params[3*i+2], i)
        for i in range(n_qubits-1):
            qc.crx(ent_params[i], i, i+1)
        # No measurement; the circuit will be used inside EstimatorQNN
        return qc

    def run(self, input_values: list[float], weight_values: list[float]) -> np.ndarray:
        """Convenience wrapper delegating to the underlying EstimatorQNN."""
        return self.qnn(input_values, weight_values)
