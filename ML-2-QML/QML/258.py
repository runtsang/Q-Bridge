"""Hybrid quantum circuit with multiple parameterised layers.

The quantum module expands the original QFCModel by stacking several
variational layers, each consisting of single‑qubit rotations followed
by a ring of controlled‑X gates.  A random layer injects additional
expressivity and the measurement is followed by a batch‑norm to
stabilise training.  The design keeps the same 4‑wire output but
offers a richer ansatz that can be tuned via the `n_layers` argument.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf

class QFCModel(tq.QuantumModule):
    """Quantum fully‑connected model with multiple variational layers.

    Parameters
    ----------
    n_wires : int, default 4
        Number of qubits used in the circuit.
    encoder_name : str, default '4x4_ryzxy'
        Name of the pretrained encoder from torchquantum.
    n_layers : int, default 3
        Number of variational layers applied after the encoder.
    use_random_layer : bool, default True
        Whether to prepend a random layer to the circuit.
    """

    class QLayer(tq.QuantumModule):
        """Single variational layer: rotations + entangling CRX ring."""
        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            self.rx = [tq.RX(has_params=True, trainable=True) for _ in range(n_wires)]
            self.ry = [tq.RY(has_params=True, trainable=True) for _ in range(n_wires)]
            self.rz = [tq.RZ(has_params=True, trainable=True) for _ in range(n_wires)]
            # Entangling CRX between consecutive wires in a ring
            self.crx = [tq.CRX(has_params=True, trainable=True) for _ in range(n_wires)]

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice):
            for i in range(self.n_wires):
                self.rx[i](qdev, wires=i)
                self.ry[i](qdev, wires=i)
                self.rz[i](qdev, wires=i)
            for i in range(self.n_wires):
                self.crx[i](qdev, wires=[i, (i + 1) % self.n_wires])

    def __init__(
        self,
        n_wires: int = 4,
        encoder_name: str = "4x4_ryzxy",
        n_layers: int = 3,
        use_random_layer: bool = True,
    ) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[encoder_name])
        self.use_random_layer = use_random_layer
        if self.use_random_layer:
            self.random_layer = tq.RandomLayer(n_ops=50, wires=list(range(self.n_wires)))
        self.q_layers = nn.ModuleList([self.QLayer(self.n_wires) for _ in range(n_layers)])
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(self.n_wires)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, 1, H, W).

        Returns
        -------
        torch.Tensor
            Normalised measurement expectation values of shape (B, n_wires).
        """
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True)
        pooled = F.avg_pool2d(x, 6).view(bsz, 16)
        self.encoder(qdev, pooled)
        if self.use_random_layer:
            self.random_layer(qdev)
        for layer in self.q_layers:
            layer(qdev)
        out = self.measure(qdev)
        return self.norm(out)

__all__ = ["QFCModel"]
