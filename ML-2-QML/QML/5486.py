import torch
import torch.nn as nn
import numpy as np
import torchquantum as tq
from torchquantum.functional import func_name_dict

# Quantum‑kernel implementation from the seed repository
from.QuantumKernelMethod import KernalAnsatz

class HybridNAT(tq.QuantumModule):
    """
    Quantum counterpart of :class:`HybridNAT`.

    Combines a classical data encoder, a variational QLayer, a QCNN‑style ansatz,
    and a fixed quantum kernel.  The forward pass accepts a batch of images,
    encodes them, processes through the variational circuit, and returns the
    measurement of Pauli‑Z on all wires.  The ``kernel`` method evaluates
    the inner‑product kernel between two batches of data.
    """

    class QLayer(tq.QuantumModule):
        def __init__(self):
            super().__init__()
            self.n_wires = 4
            self.random_layer = tq.RandomLayer(n_ops=50, wires=list(range(self.n_wires)))
            self.rx0 = tq.RX(has_params=True, trainable=True)
            self.ry0 = tq.RY(has_params=True, trainable=True)
            self.rz0 = tq.RZ(has_params=True, trainable=True)
            self.crx0 = tq.CRX(has_params=True, trainable=True)

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice):
            self.random_layer(qdev)
            self.rx0(qdev, wires=0)
            self.ry0(qdev, wires=1)
            self.rz0(qdev, wires=3)
            self.crx0(qdev, wires=[0, 2])
            func_name_dict["hadamard"](qdev, wires=3)
            func_name_dict["sx"](qdev, wires=2)
            func_name_dict["cnot"](qdev, wires=[3, 0])

    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.q_layer = self.QLayer()
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(self.n_wires)

        # QCNN‑style ansatz (constructed once for reuse)
        self.qcnn_ansatz = self._build_qcnn_ansatz()

        # Quantum kernel ansatz
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.kernel_ansatz = KernalAnsatz(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )

    def _build_qcnn_ansatz(self):
        """
        Build a QCNN‑style ansatz using Qiskit primitives.  The circuit is
        decomposed to a unitary that can be used later for kernel evaluation
        or as part of a larger variational circuit.
        """
        from qiskit import QuantumCircuit
        from qiskit.circuit import ParameterVector

        def conv_circuit(params):
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

        def conv_layer(num_qubits, param_prefix):
            qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
            qubits = list(range(num_qubits))
            param_index = 0
            params = ParameterVector(param_prefix, length=num_qubits * 3)
            for q1, q2 in zip(qubits[0::2], qubits[1::2]):
                qc = qc.compose(conv_circuit(params[param_index : param_index + 3]), [q1, q2])
                qc.barrier()
                param_index += 3
            for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
                qc = qc.compose(conv_circuit(params[param_index : param_index + 3]), [q1, q2])
                qc.barrier()
                param_index += 3
            return qc

        def pool_circuit(params):
            qc = QuantumCircuit(2)
            qc.rz(-np.pi / 2, 1)
            qc.cx(1, 0)
            qc.rz(params[0], 0)
            qc.ry(params[1], 1)
            qc.cx(0, 1)
            qc.ry(params[2], 1)
            return qc

        def pool_layer(sources, sinks, param_prefix):
            num_qubits = len(sources) + len(sinks)
            qc = QuantumCircuit(num_qubits, name="Pooling Layer")
            param_index = 0
            params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
            for source, sink in zip(sources, sinks):
                qc = qc.compose(pool_circuit(params[param_index : param_index + 3]), [source, sink])
                qc.barrier()
                param_index += 3
            return qc

        ansatz = QuantumCircuit(8, name="Ansatz")
        ansatz.compose(conv_layer(8, "c1"), list(range(8)), inplace=True)
        ansatz.compose(pool_layer([0, 1, 2, 3], [4, 5, 6, 7], "p1"), list(range(8)), inplace=True)
        ansatz.compose(conv_layer(4, "c2"), list(range(4, 8)), inplace=True)
        ansatz.compose(pool_layer([0, 1], [2, 3], "p2"), list(range(4, 8)), inplace=True)
        ansatz.compose(conv_layer(2, "c3"), list(range(6, 8)), inplace=True)
        ansatz.compose(pool_layer([0], [1], "p3"), list(range(6, 8)), inplace=True)
        return ansatz

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Execute the hybrid quantum circuit.

        Parameters
        ----------
        x : torch.Tensor
            Batch of images with shape (batch, 1, H, W).

        Returns
        -------
        torch.Tensor
            Batch of normalized Pauli‑Z measurement results.
        """
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(
            n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True
        )
        pooled = torch.nn.functional.avg_pool2d(x, 6).view(bsz, 16)
        self.encoder(qdev, pooled)
        self.q_layer(qdev)
        out = self.measure(qdev)
        return self.norm(out)

    def kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the quantum kernel between two batches.

        Parameters
        ----------
        x, y : torch.Tensor
            Input batches of shape (batch, features).  For simplicity the
            implementation reshapes to (1, -1).

        Returns
        -------
        torch.Tensor
            Kernel value (a scalar).
        """
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.kernel_ansatz(self.q_device, x, y)
        return torch.abs(self.q_device.states.view(-1)[0])
