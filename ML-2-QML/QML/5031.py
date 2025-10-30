import torch
import torch.nn as nn
import torchquantum as tq
from torchquantum import QuantumDevice
from torchquantum.functional import func_name_dict

class HybridKernelModel(tq.QuantumModule):
    """Quantum kernel with a QCNN‑style variational ansatz.

    The kernel is the squared magnitude of the overlap between two
    variational states.  The ansatz comprises a random layer, a set of
    RX/RY rotations parameterised by a depth‑wise tensor, and
    optional QCNN‑style convolution and pooling blocks that mimic the
    convolution‑pooling pattern from the QCNN reference.

    Parameters
    ----------
    n_wires : int, default 4
        Number of qubits.
    depth : int, default 2
        Number of variational blocks.
    use_qcnn : bool, default False
        If True, a QCNN‑style block is inserted after the random layer.
    """

    def __init__(
        self,
        n_wires: int = 4,
        depth: int = 2,
        use_qcnn: bool = False,
    ) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.depth = depth
        self.use_qcnn = use_qcnn

        # Device used for all circuits
        self.q_device = QuantumDevice(n_wires=self.n_wires)

        # Random feature layer
        self.random = tq.RandomLayer(n_ops=20, wires=list(range(self.n_wires)))

        # Variational parameters: (depth, n_wires, 2) for RX and RY
        self.params = nn.Parameter(
            torch.randn(self.depth, self.n_wires, 2, dtype=torch.float32)
        )

        # Optional QCNN blocks
        self.blocks = nn.ModuleList()
        if self.use_qcnn:
            for _ in range(self.depth):
                self.blocks.append(self._build_qcnn_block())

    def _build_qcnn_block(self) -> nn.Module:
        """Return a single QCNN convolution‑pooling block."""
        block = nn.ModuleList()
        # Convolution
        block.append(tq.RZ(has_params=True, trainable=True))
        block.append(tq.CX())
        block.append(tq.RY(has_params=True, trainable=True))
        # Pooling
        block.append(tq.RZ(has_params=True, trainable=True))
        block.append(tq.CX())
        block.append(tq.RY(has_params=True, trainable=True))
        return block

    def _apply_block(
        self,
        qdev: QuantumDevice,
        block: nn.ModuleList,
        params: torch.Tensor,
    ) -> None:
        """Apply a QCNN block with given parameters."""
        for op, param in zip(block, params):
            if isinstance(op, (tq.RZ, tq.RY)):
                op(qdev, wires=list(range(self.n_wires)), params=param)
            else:
                op(qdev, wires=list(range(self.n_wires)))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute the quantum kernel value between *x* and *y*."""
        # Reshape to (batch, n_wires)
        x = x.reshape(-1, self.n_wires)
        y = y.reshape(-1, self.n_wires)

        # Reset device for combined batch
        self.q_device.reset_states(x.shape[0] + y.shape[0])

        # Encode first batch
        for wire in range(self.n_wires):
            self.q_device.rx(x[:, wire], wires=[wire])
        self.random(self.q_device)
        for idx in range(self.depth):
            if self.use_qcnn:
                self._apply_block(
                    self.q_device, self.blocks[idx], self.params[idx]
                )
            else:
                self.q_device.rx(self.params[idx, :, 0], wires=list(range(self.n_wires)))
                self.q_device.ry(self.params[idx, :, 1], wires=list(range(self.n_wires)))

        # Encode second batch with reversed parameters
        for wire in range(self.n_wires):
            self.q_device.rx(-y[:, wire], wires=[wire])
        self.random(self.q_device)
        for idx in range(self.depth):
            if self.use_qcnn:
                self._apply_block(
                    self.q_device, self.blocks[idx], -self.params[idx]
                )
            else:
                self.q_device.rx(-self.params[idx, :, 0], wires=list(range(self.n_wires)))
                self.q_device.ry(-self.params[idx, :, 1], wires=list(range(self.n_wires)))

        # Overlap (squared magnitude of the first amplitude)
        overlap = torch.abs(self.q_device.states.view(-1)[0])
        return overlap

    def kernel_matrix(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Return the Gram matrix for two batches of samples."""
        a = a.reshape(-1, self.n_wires)
        b = b.reshape(-1, self.n_wires)
        n_a, n_b = a.size(0), b.size(0)
        kernel = torch.empty((n_a, n_b), device=a.device)
        for i in range(n_a):
            kernel[i] = self.forward(a[i].unsqueeze(0), b)
        return kernel

__all__ = ["HybridKernelModel"]
