from __future__ import annotations

from typing import List, Tuple
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN as QiskitEstimatorQNN
from qiskit.primitives import StatevectorEstimator

class QuantumEstimatorQNN:
    """Quantum circuit for regression that accepts 2 input parameters and
    one trainable weight parameter per qubit, measuring Y on the first qubit."""
    def __init__(self, n_qubits: int = 2) -> None:
        self.n_qubits = n_qubits
        self.input_params = [Parameter(f"in_{i}") for i in range(n_qubits)]
        self.weight_params = [Parameter(f"w_{i}") for i in range(n_qubits)]
        self.qc = self._build_circuit()
        self.observable = SparsePauliOp.from_list([("Y" + "I" * (n_qubits - 1), 1)])
        self.estimator = StatevectorEstimator()
        self.estimator_qnn = QiskitEstimatorQNN(
            circuit=self.qc,
            observables=self.observable,
            input_params=self.input_params,
            weight_params=self.weight_params,
            estimator=self.estimator,
        )

    def _build_circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.n_qubits)
        qc.h(0)
        for i in range(self.n_qubits):
            qc.ry(self.input_params[i], i)
            qc.rx(self.weight_params[i], i)
        return qc

class QLSTM:
    """Quantum‑enhanced LSTM where each gate is a parameterised circuit.
    The hidden state is encoded in the first qubit; the next hidden state
    is read from expectation of Pauli‑Z on that qubit."""
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 n_qubits: int = 4) -> None:
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.gate_circuits = {
            "forget": self._gate_circuit(),
            "input": self._gate_circuit(),
            "update": self._gate_circuit(),
            "output": self._gate_circuit(),
        }
        self.weight_params = {k: [Parameter(f"{k}_{i}") for i in range(n_qubits)]
                              for k in self.gate_circuits}
        self.estimator = StatevectorEstimator()
        self.meas_op = SparsePauliOp.from_list([("Z" + "I" * (n_qubits - 1), 1)])

    def _gate_circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.n_qubits)
        for i in range(self.n_qubits):
            qc.rx(Parameter(f"c_{i}"), i)
        for i in range(self.n_qubits - 1):
            qc.cx(i, i + 1)
        return qc

    def forward(self,
                inputs: List[float],
                hidden_state: float = 0.0) -> Tuple[float, float]:
        """Apply the four gate circuits sequentially and return the new hidden
        state as the Z expectation of the first qubit and a placeholder cell
        state (here identical to hidden)."""
        h = hidden_state
        for gate in ["forget", "input", "update", "output"]:
            qc = self.gate_circuits[gate]
            # bind weight parameters to current hidden state value
            param_bind = {p: h for p in self.weight_params[gate]}
            # bind input parameters to current input
            param_bind.update({p: inputs[0] for p in qc.parameters})
            qc_bound = qc.bind_parameters(param_bind)
            meas = self.estimator.run(circuits=qc_bound,
                                      observables=[self.meas_op]).result().values[0]
            h = meas.real  # use real part as new hidden
        return h, h  # placeholder for cell state

__all__ = ["QuantumEstimatorQNN", "QLSTM"]
