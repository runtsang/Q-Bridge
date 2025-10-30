import torch
import torch.nn as nn
import numpy as np
import torchquantum as tq
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import StatevectorEstimator as Estimator

class QuantumLayer(tq.QuantumModule):
    """QCNN‑style variational quantum layer."""
    def __init__(self, n_wires: int = 4):
        super().__init__()
        self.n_wires = n_wires
        # Encoder: simple universal single‑qubit gate set
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        # Convolution and pooling parameters: 3 per pair of qubits
        self.conv_params = nn.ParameterList([nn.Parameter(torch.randn(3)) for _ in range(self.n_wires // 2)])
        self.pool_params = nn.ParameterList([nn.Parameter(torch.randn(3)) for _ in range(self.n_wires // 2)])
        self.measure = tq.MeasureAll(tq.PauliZ)

    @staticmethod
    def _conv_block(qdev, wires, params):
        """Two‑qubit convolution operation."""
        qdev.rz(-np.pi / 2, wires[1])
        qdev.cx(wires[1], wires[0])
        qdev.rz(params[0], wires[0])
        qdev.ry(params[1], wires[1])
        qdev.cx(wires[0], wires[1])
        qdev.ry(params[2], wires[1])
        qdev.cx(wires[1], wires[0])
        qdev.rz(np.pi / 2, wires[0])

    @staticmethod
    def _pool_block(qdev, wires, params):
        """Pooling operation that reduces two qubits to one."""
        qdev.rz(-np.pi / 2, wires[1])
        qdev.cx(wires[1], wires[0])
        qdev.rz(params[0], wires[0])
        qdev.ry(params[1], wires[1])
        qdev.cx(wires[0], wires[1])
        qdev.ry(params[2], wires[1])

    @tq.static_support
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Classical feature tensor of shape (batch, n_wires).
        Returns
        -------
        torch.Tensor
            Measurement outcomes of shape (batch, n_wires).
        """
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device)
        # Encode classical data
        self.encoder(qdev, x)
        # Convolution on qubit pairs (0,1) and (2,3)
        self._conv_block(qdev, [0, 1], self.conv_params[0])
        self._conv_block(qdev, [2, 3], self.conv_params[1])
        # Pooling on the same pairs
        self._pool_block(qdev, [0, 1], self.pool_params[0])
        self._pool_block(qdev, [2, 3], self.pool_params[1])
        # Measurement
        return self.measure(qdev)

    def get_qiskit_estimator(self):
        """
        Build a Qiskit EstimatorQNN that mirrors the variational circuit.
        Returns
        -------
        EstimatorQNN
            Qiskit variational circuit ready for state‑vector or real‑device execution.
        """
        qc = QuantumCircuit(self.n_wires)
        # Feature map: simple Hadamard on each qubit
        for i in range(self.n_wires):
            qc.h(i)
        # Parameter vector for all variational gates
        param_vec = ParameterVector("θ", length=self.n_wires)
        # Convolution blocks
        for i in range(0, self.n_wires, 2):
            idx = i // 2
            conv = self._build_conv_circuit(param_vec[3 * idx: 3 * idx + 3], [i, i + 1])
            qc.append(conv, [i, i + 1])
        # Pooling blocks
        for i in range(0, self.n_wires, 2):
            idx = i // 2 + self.n_wires // 2
            pool = self._build_pool_circuit(param_vec[3 * idx: 3 * idx + 3], [i, i + 1])
            qc.append(pool, [i, i + 1])
        # Observable: Z on all qubits
        observable = SparsePauliOp.from_list([("Z" * self.n_wires, 1)])
        estimator = Estimator()
        return EstimatorQNN(
            circuit=qc,
            observables=observable,
            input_params=[],
            weight_params=param_vec,
            estimator=estimator
        )

    @staticmethod
    def _build_conv_circuit(params, wires):
        circ = QuantumCircuit(2)
        circ.rz(-np.pi / 2, 1)
        circ.cx(1, 0)
        circ.rz(params[0], 0)
        circ.ry(params[1], 1)
        circ.cx(0, 1)
        circ.ry(params[2], 1)
        circ.cx(1, 0)
        circ.rz(np.pi / 2, 0)
        return circ

    @staticmethod
    def _build_pool_circuit(params, wires):
        circ = QuantumCircuit(2)
        circ.rz(-np.pi / 2, 1)
        circ.cx(1, 0)
        circ.rz(params[0], 0)
        circ.ry(params[1], 1)
        circ.cx(0, 1)
        circ.ry(params[2], 1)
        return circ

__all__ = ["QuantumLayer"]
