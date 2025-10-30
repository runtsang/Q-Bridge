"""Quantum self‑attention with quanvolution‑style convolution (TorchQuantum)."""

import torch
import torchquantum as tq
import torchquantum.circuit as tqc


class QuantumSelfAttentionQuanvolution(tqc.QuantumModule):
    """
    Variational circuit that jointly encodes a 2×2 image patch into
    a quantum kernel for self‑attention and a separate kernel for
    quanvolution.  The circuit operates on 6 qubits:
        - qubits 0–1 encode the attention query/key
        - qubits 2–5 encode the quanvolution kernel
    """
    def __init__(self,
                 n_wires: int = 6,
                 n_ops_attention: int = 4,
                 n_ops_quanvolution: int = 4) -> None:
        super().__init__()
        self.n_wires = n_wires

        # Attention encoder (first 2 qubits)
        self.att_encoder = tqc.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
            ]
        )
        self.att_layer = tqc.RandomLayer(n_ops=n_ops_attention, wires=[0, 1])

        # Quanvolution encoder (next 4 qubits)
        self.q_encoder = tqc.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [2]},
                {"input_idx": [1], "func": "ry", "wires": [3]},
                {"input_idx": [2], "func": "ry", "wires": [4]},
                {"input_idx": [3], "func": "ry", "wires": [5]},
            ]
        )
        self.q_layer = tqc.RandomLayer(n_ops=n_ops_quanvolution, wires=[2, 3, 4, 5])

        self.measure = tqc.MeasureAll(tqc.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input image batch of shape (B, 1, 28, 28).

        Returns
        -------
        torch.Tensor
            Concatenated measurement vector of shape (B, 6 * 14 * 14).
        """
        bsz = x.shape[0]
        device = x.device
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)

        x = x.view(bsz, 28, 28)
        patches = []

        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                # 2×2 patch flattened into 4 values
                data = torch.stack(
                    [
                        x[:, r, c],
                        x[:, r, c + 1],
                        x[:, r + 1, c],
                        x[:, r + 1, c + 1],
                    ],
                    dim=1,
                )

                # Encode attention part (first 2 values)
                self.att_encoder(qdev, data[:, :2])
                self.att_layer(qdev)

                # Encode quanvolution part (remaining 2 values)
                self.q_encoder(qdev, data[:, 2:])
                self.q_layer(qdev)

                measurement = self.measure(qdev)
                patches.append(measurement.view(bsz, self.n_wires))

        return torch.cat(patches, dim=1)


def SelfAttention() -> QuantumSelfAttentionQuanvolution:
    """
    Factory mirroring the original API.
    Returns an instance of the hybrid quantum module.
    """
    return QuantumSelfAttentionQuanvolution()


__all__ = ["QuantumSelfAttentionQuanvolution", "SelfAttention"]
