import torch
import torchquantum as tq
import torchquantum.functional as tqf


class FraudDetectionHybrid:
    """
    Quantum‑augmented fraud‑detection model.

    The class implements a parameterised quantum encoder followed by
    a few variational layers and a classical read‑out head.  It can be
    used as a drop‑in replacement for the classical
    :class:`FraudDetectionHybrid` defined in the ML module.

    Parameters
    ----------
    num_wires : int
        Number of qubits used for encoding.
    """

    def __init__(self, num_wires: int = 2) -> None:
        self.num_wires = num_wires
        # Encoder that maps classical inputs to quantum states
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "rx", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
            ]
        )
        # Variational layer (simple RX + RY per wire)
        self.var_layer = tq.QuantumModule(
            tq.RX(has_params=True, trainable=True),
            tq.RY(has_params=True, trainable=True),
        )
        # Measurement
        self.measure = tq.MeasureAll(tq.PauliZ)

        # Classical classifier head
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(num_wires, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 1),
            torch.nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Tensor of shape (batch, 2) where each row contains
            the two classical fraud features.

        Returns
        -------
        torch.Tensor
            Fraud probability of shape (batch,).
        """
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.num_wires, bsz=bsz, device=x.device)

        # Encode classical data
        self.encoder(qdev, x)

        # Variational circuit
        self.var_layer(qdev)

        # Measure expectation values
        out = self.measure(qdev)

        # Classical read‑out
        return self.classifier(out).squeeze(-1)


__all__ = ["FraudDetectionHybrid"]
