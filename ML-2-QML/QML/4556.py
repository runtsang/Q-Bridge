"""Quantum implementation of a hybrid layer combining fully‑connected, attention, LSTM‑style gating, and classification.

The class uses Qiskit to build parameterized circuits for each sub‑module and measures expectation values of Pauli‑Z on the first qubit to obtain gate activations and logits.  This design enables end‑to‑end quantum‑classical experiments while keeping the interface identical to the classical counterpart.
"""

from __future__ import annotations

from typing import Dict, Iterable, Tuple
import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute, Aer
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp


class HybridLayer:
    """
    Quantum hybrid layer that emulates the classical HybridLayer.
    """

    def __init__(
        self,
        n_qubits: int,
        depth: int,
        use_attention: bool = False,
        use_lstm_gate: bool = False,
        classifier_depth: int = 0,
    ) -> None:
        self.n_qubits = n_qubits
        self.depth = depth
        self.use_attention = use_attention
        self.use_lstm_gate = use_lstm_gate
        self.classifier_depth = classifier_depth

        # Backend
        self.backend = Aer.get_backend("qasm_simulator")
        self.shots = 1024

        # Fully connected circuit
        self.fc_circuit, self.fc_params = self._build_fc_circuit()

        # Attention circuit
        if self.use_attention:
            self.attn_circuit, self.attn_params = self._build_attention_circuit()

        # LSTM gate circuits
        if self.use_lstm_gate:
            self.forget_circuit, self.forget_params = self._build_gate_circuit("forget")
            self.input_circuit, self.input_params = self._build_gate_circuit("input")
            self.update_circuit, self.update_params = self._build_gate_circuit("update")
            self.output_circuit, self.output_params = self._build_gate_circuit("output")

        # Classifier circuit
        if self.classifier_depth > 0:
            self.classifier_circuit, self.classifier_params = self._build_classifier_circuit()
        else:
            self.classifier_circuit = None

    def _build_fc_circuit(self) -> Tuple[QuantumCircuit, Iterable]:
        """Build a simple layered ansatz for the fully‑connected block."""
        encoding = ParameterVector("x", self.n_qubits)
        weights = ParameterVector("theta_fc", self.n_qubits * self.depth)

        qc = QuantumCircuit(self.n_qubits)
        for i, param in enumerate(encoding):
            qc.rx(param, i)

        idx = 0
        for _ in range(self.depth):
            for i in range(self.n_qubits):
                qc.ry(weights[idx], i)
                idx += 1
            for i in range(self.n_qubits - 1):
                qc.cz(i, i + 1)

        return qc, list(encoding) + list(weights)

    def _build_attention_circuit(self) -> Tuple[QuantumCircuit, Iterable]:
        """Quantum self‑attention block."""
        rot = ParameterVector("rot", self.n_qubits * 3)
        ent = ParameterVector("ent", self.n_qubits - 1)

        qc = QuantumCircuit(self.n_qubits)
        for i in range(self.n_qubits):
            qc.rx(rot[3 * i], i)
            qc.ry(rot[3 * i + 1], i)
            qc.rz(rot[3 * i + 2], i)

        for i in range(self.n_qubits - 1):
            qc.crx(ent[i], i, i + 1)

        return qc, list(rot) + list(ent)

    def _build_gate_circuit(self, name: str) -> Tuple[QuantumCircuit, Iterable]:
        """Build a small circuit that outputs a single expectation value for a gate."""
        gate_params = ParameterVector(f"{name}_theta", self.n_qubits)
        qc = QuantumCircuit(self.n_qubits)
        for i in range(self.n_qubits):
            qc.rx(gate_params[i], i)
        # Measure first qubit in Z basis
        qc.measure(0, 0)
        return qc, list(gate_params)

    def _build_classifier_circuit(self) -> Tuple[QuantumCircuit, Iterable]:
        """Build a layered ansatz for the classification head."""
        encoding = ParameterVector("x_cls", self.n_qubits)
        weights = ParameterVector("theta_cls", self.n_qubits * self.depth)

        qc = QuantumCircuit(self.n_qubits)
        for i, param in enumerate(encoding):
            qc.rx(param, i)

        idx = 0
        for _ in range(self.depth):
            for i in range(self.n_qubits):
                qc.ry(weights[idx], i)
                idx += 1
            for i in range(self.n_qubits - 1):
                qc.cz(i, i + 1)

        # Final measurement on all qubits
        qc.measure_all()
        return qc, list(encoding) + list(weights)

    def _expectation(self, circuit: QuantumCircuit, param_binds: Dict) -> float:
        """Run circuit and return expectation value of Pauli‑Z on qubit 0."""
        job = execute(circuit, self.backend, shots=self.shots, parameter_binds=[param_binds])
        result = job.result()
        counts = result.get_counts(circuit)
        # Convert counts to expectation
        exp = 0.0
        for bitstring, cnt in counts.items():
            z = 1 if bitstring[-1] == "0" else -1  # qubit 0 is last in bitstring
            exp += z * cnt
        return exp / self.shots

    def run(self, input_sequence: np.ndarray, param_dict: Dict[str, Iterable]) -> np.ndarray:
        """
        Execute the hybrid quantum layer.

        Parameters
        ----------
        input_sequence : np.ndarray
            Input data of shape (seq_len, n_qubits) with values in [-π, π].
        param_dict : dict
            Mapping from parameter names to iterable values.

        Returns
        -------
        np.ndarray
            Logits from the classifier head (if enabled) or final hidden state.
        """
        seq_len = input_sequence.shape[0]
        hx = np.zeros(self.n_qubits)
        cx = np.zeros(self.n_qubits)

        for step in range(seq_len):
            # Encode input for this step
            step_params = {f"x[{i}]": input_sequence[step, i] for i in range(self.n_qubits)}
            # Fully connected output
            fc_params = {f"x[{i}]": input_sequence[step, i] for i in range(self.n_qubits)}
            fc_params.update({f"theta_fc[{i}]": param_dict[f"theta_fc[{i}]"] for i in range(self.n_qubits * self.depth)})

            fc_expect = self._expectation(self.fc_circuit, fc_params)

            # Attention (if enabled)
            if self.use_attention:
                attn_params = {f"rot[{i}]": param_dict[f"rot[{i}]"] for i in range(self.n_qubits * 3)}
                attn_params.update({f"ent[{i}]": param_dict[f"ent[{i}]"] for i in range(self.n_qubits - 1)})
                attn_expect = self._expectation(self.attn_circuit, attn_params)
            else:
                attn_expect = 0.0

            # LSTM gates (if enabled)
            if self.use_lstm_gate:
                gate_expect = {}
                for gate_name, (gate_circ, gate_params) in {
                    "forget": (self.forget_circuit, self.forget_params),
                    "input": (self.input_circuit, self.input_params),
                    "update": (self.update_circuit, self.update_params),
                    "output": (self.output_circuit, self.output_params),
                }.items():
                    gate_param_bind = {f"{gate_name}_theta[{i}]": param_dict[f"{gate_name}_theta[{i}]"] for i in range(self.n_qubits)}
                    gate_expect[gate_name] = self._expectation(gate_circ, gate_param_bind)

                # Map expectation values to gate activations
                f = 1 / (1 + np.exp(-gate_expect["forget"]))
                i = 1 / (1 + np.exp(-gate_expect["input"]))
                g = np.tanh(gate_expect["update"])
                o = 1 / (1 + np.exp(-gate_expect["output"]))

                cx = f * cx + i * g
                hx = o * np.tanh(cx)
            else:
                hx = np.array([fc_expect, attn_expect])  # simple placeholder

        # Classifier (if enabled)
        if self.classifier_circuit is not None:
            cls_params = {f"x_cls[{i}]": hx[i] for i in range(self.n_qubits)}
            cls_params.update({f"theta_cls[{i}]": param_dict[f"theta_cls[{i}]"] for i in range(self.n_qubits * self.depth)})
            logits = self._expectation(self.classifier_circuit, cls_params)
            return np.array([logits])
        else:
            return hx

__all__ = ["HybridLayer"]
