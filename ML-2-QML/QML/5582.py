"""Quantum hybrid QCNN‑Autoencoder‑LSTM circuit.

The circuit implements:
* A Z‑feature map for input encoding.
* Convolution and pooling blocks with 3‑parameter gates per qubit pair.
* A simplified quantum auto‑encoder using a swap‑test style sub‑circuit.
* A fully‑connected layer realised by parameterised rotations.
* A quantum‑LSTM block (placeholder) using RX rotations.
The circuit outputs the expectation value of a Z observable on the first
input qubits.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Parameter, ParameterVector
from qiskit.circuit.library import ZFeatureMap, RealAmplitudes
from qiskit.primitives import Estimator
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN

class HybridQCNNQML:
    """Quantum circuit that mirrors the HybridQCNNModel architecture."""

    def __init__(
        self,
        input_dim: int = 8,
        latent_dim: int = 3,
        lstm_qubits: int = 4,
        seed: int = 42,
    ) -> None:
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.lstm_qubits = lstm_qubits
        self.estimator = Estimator()
        self.circuit, self.input_params, self.weight_params = self._build_circuit()
        self.qnn = EstimatorQNN(
            circuit=self.circuit,
            input_params=self.input_params,
            weight_params=self.weight_params,
            estimator=self.estimator,
            observables=SparsePauliOp.from_list([("Z" * input_dim, 1)]),
        )

    def _build_circuit(self) -> tuple[QuantumCircuit, list[Parameter], list[Parameter]]:
        """Construct the full QCNN‑auto‑encoder‑LSTM circuit."""
        # Feature map
        feature_map = ZFeatureMap(self.input_dim)

        # Convolution layer
        conv = self._conv_layer(self.input_dim)

        # Pooling layer
        pool = self._pool_layer(self.input_dim)

        # Auto‑encoder sub‑circuit
        autoenc = self._autoencoder_circuit(self.latent_dim)

        # Fully‑connected layer
        fcl = self._fcl_circuit(self.latent_dim)

        # Quantum‑LSTM block
        lstm = self._lstm_circuit(self.lstm_qubits)

        # Assemble
        total_qubits = (
            self.input_dim
            + 2 * self.latent_dim
            + 1
            + self.lstm_qubits
        )
        qc = QuantumCircuit(total_qubits)

        # Map qubit ranges
        idx = 0
        # Feature map, conv, pool on first input_dim qubits
        qc.compose(feature_map, qargs=range(idx, idx + self.input_dim), inplace=True)
        idx += self.input_dim
        qc.compose(conv, qargs=range(idx - self.input_dim, idx), inplace=True)
        qc.compose(pool, qargs=range(idx - self.input_dim, idx), inplace=True)

        # Auto‑encoder on next 2*latent_dim+1 qubits
        auto_start = idx
        qc.compose(autoenc, qargs=range(auto_start, auto_start + 2 * self.latent_dim + 1), inplace=True)
        idx = auto_start + 2 * self.latent_dim + 1

        # Fully‑connected on same auto‑encoder qubits
        qc.compose(fcl, qargs=range(auto_start, idx), inplace=True)

        # LSTM on final qubits
        lstm_start = idx
        qc.compose(lstm, qargs=range(lstm_start, lstm_start + self.lstm_qubits), inplace=True)

        # Measurement of all qubits for expectation
        qc.measure_all()
        # Gather parameters
        input_params = list(feature_map.parameters)
        weight_params = (
            list(conv.parameters)
            + list(pool.parameters)
            + list(autoenc.parameters)
            + list(fcl.parameters)
            + list(lstm.parameters)
        )
        return qc, input_params, weight_params

    def _conv_layer(self, num_qubits: int) -> QuantumCircuit:
        """Create a convolution block with 3‑parameter gates per pair."""
        params = ParameterVector("c", length=3 * num_qubits)
        qc = QuantumCircuit(num_qubits)
        for q in range(0, num_qubits, 2):
            sub = self._conv_circuit(params[q * 3 : (q + 2) * 3])
            qc.append(sub, [q, q + 1])
        return qc

    def _conv_circuit(self, params: list[Parameter]) -> QuantumCircuit:
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

    def _pool_layer(self, num_qubits: int) -> QuantumCircuit:
        """Pooling sub‑circuit with 3 parameters per pair."""
        params = ParameterVector("p", length=3 * (num_qubits // 2))
        qc = QuantumCircuit(num_qubits)
        for q in range(0, num_qubits, 2):
            sub = self._pool_circuit(params[(q // 2) * 3 : ((q // 2) + 1) * 3])
            qc.append(sub, [q, q + 1])
        return qc

    def _pool_circuit(self, params: list[Parameter]) -> QuantumCircuit:
        qc = QuantumCircuit(2)
        qc.rz(-np.pi / 2, 1)
        qc.cx(1, 0)
        qc.rz(params[0], 0)
        qc.ry(params[1], 1)
        qc.cx(0, 1)
        qc.ry(params[2], 1)
        return qc

    def _autoencoder_circuit(self, latent_dim: int) -> QuantumCircuit:
        """Simplified quantum auto‑encoder using a swap‑test style."""
        num_trash = 2
        qr = QuantumRegister(latent_dim + 2 * num_trash + 1, "q")
        cr = ClassicalRegister(1, "c")
        qc = QuantumCircuit(qr, cr)
        qc.h(range(latent_dim + num_trash))
        aux = latent_dim + 2 * num_trash
        qc.h(aux)
        for i in range(num_trash):
            qc.cswap(aux, latent_dim + i, latent_dim + num_trash + i)
        qc.h(aux)
        qc.measure(aux, cr[0])
        return qc

    def _fcl_circuit(self, latent_dim: int) -> QuantumCircuit:
        """Parameterised rotations acting as a fully‑connected layer."""
        total = 2 * latent_dim + 1
        params = ParameterVector("f", length=total)
        qc = QuantumCircuit(total)
        for i in range(total):
            qc.ry(params[i], i)
        return qc

    def _lstm_circuit(self, n_qubits: int) -> QuantumCircuit:
        """Placeholder quantum LSTM block using RX rotations."""
        params = ParameterVector("l", length=n_qubits)
        qc = QuantumCircuit(n_qubits)
        for i in range(n_qubits):
            qc.rx(params[i], i)
        return qc

    def run(self, inputs: np.ndarray, shots: int = 1024) -> np.ndarray:
        """Execute the circuit for a batch of input feature vectors."""
        # Bind feature map parameters
        bind = {p: val for p, val in zip(self.input_params, inputs.T)}
        job = self.estimator.run(self.circuit, shots=shots, parameter_binds=[bind])
        result = job.result()
        return result.get_expectation_value()

__all__ = ["HybridQCNNQML"]
