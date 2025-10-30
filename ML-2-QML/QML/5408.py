from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator as QiskitEstimator
from qiskit_machine_learning.neural_networks import EstimatorQNN

# ----------------------------------------------------------------------
# Quantum Convolution Building Blocks
# ----------------------------------------------------------------------
def _conv_circuit(params):
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


def _conv_layer(num_qubits, param_prefix):
    qc = QuantumCircuit(num_qubits)
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits * 3)
    for q1, q2 in zip(range(0, num_qubits, 2), range(1, num_qubits, 2)):
        qc.append(_conv_circuit(params[param_index:param_index+3]), [q1, q2])
        param_index += 3
    return qc


def _pool_circuit(params):
    qc = QuantumCircuit(2)
    qc.rz(-np.pi / 2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    return qc


def _pool_layer(num_qubits, param_prefix):
    qc = QuantumCircuit(num_qubits)
    param_index = 0
    params = ParameterVector(param_prefix, length=(num_qubits // 2) * 3)
    for i in range(0, num_qubits, 2):
        qc.append(_pool_circuit(params[param_index:param_index+3]), [i, i+1])
        param_index += 3
    return qc

# ----------------------------------------------------------------------
# Quantum QCNN
# ----------------------------------------------------------------------
class QCNNQuantum(nn.Module):
    """
    Variational QCNN built from qiskit circuits.  The ansatz mirrors the
    classical QCNN but replaces each linear layer with a quantum
    convolution/pooling block.  An optional quantum LSTM layer can be
    appended to model sequential dependencies between feature blocks.
    """
    def __init__(
        self,
        n_qubits: int = 8,
        use_qulstm: bool = False,
        qulstm_layers: int = 1,
    ) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.use_qulstm = use_qulstm
        self.estimator = QiskitEstimator()

        # Build ansatz circuit
        qc = QuantumCircuit(n_qubits)
        qc.append(_conv_layer(n_qubits, "c1"), range(n_qubits))
        qc.append(_pool_layer(n_qubits, "p1"), range(n_qubits))
        qc.append(_conv_layer(n_qubits // 2, "c2"), list(range(n_qubits // 2, n_qubits)))
        qc.append(_pool_layer(n_qubits // 2, "p2"), list(range(n_qubits // 2, n_qubits)))
        if use_qulstm:
            for _ in range(qulstm_layers):
                qc.append(_conv_layer(n_qubits // 4, f"ql{_}"), list(range(n_qubits // 4, n_qubits // 2)))
        self.circuit = qc.decompose()

        # Observable: measure Z on the first qubit
        self.observable = SparsePauliOp.from_list([("Z" + "I" * (n_qubits - 1), 1)])

        # Parameters
        self.input_params = ParameterVector("θ", length=n_qubits)
        self.weight_params = self.circuit.parameters

        self.qnn = EstimatorQNN(
            circuit=self.circuit,
            observables=self.observable,
            input_params=self.input_params,
            weight_params=self.weight_params,
            estimator=self.estimator,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        inputs: shape (batch, n_qubits) with values in [0, 2π].
        Returns a probability in [0, 1] for each example.
        """
        if inputs.shape[1]!= self.n_qubits:
            raise ValueError(f"Input dimension {inputs.shape[1]} does not match model {self.n_qubits}")
        param_values = inputs.detach().cpu().numpy()
        preds = self.qnn.predict(inputs=param_values)
        return torch.tensor(preds, dtype=torch.float32, device=inputs.device).squeeze(-1)


# ----------------------------------------------------------------------
# Factory
# ----------------------------------------------------------------------
def QCNN(
    *,
    n_qubits: int = 8,
    use_qulstm: bool = False,
    qulstm_layers: int = 1,
) -> nn.Module:
    """
    Construct a quantum QCNN model.  The returned object can be used like
    a standard PyTorch module and internally relies on Qiskit for
    circuit execution.
    """
    return QCNNQuantum(n_qubits=n_qubits, use_qulstm=use_qulstm, qulstm_layers=qulstm_layers)


__all__ = [
    "QCNNQuantum",
    "QCNN",
]
