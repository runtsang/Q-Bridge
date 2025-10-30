import torch
import torchquantum as tq

class CombinedEstimatorQNN(tq.QuantumModule):
    """
    Variational quantum circuit inspired by the EstimatorQNN example.
    The circuit encodes a 4‑dimensional classical input using RX gates,
    applies a RandomLayer and trainable RX/RY/RZ gates, and measures all qubits.
    The output is a feature vector of expectation values that can be fed into
    a classical head or used directly for regression.
    """

    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.var_layer = tq.RandomLayer(n_ops=30, wires=list(range(self.n_wires)))
        self.rx_gate = tq.RX(has_params=True, trainable=True)
        self.ry_gate = tq.RY(has_params=True, trainable=True)
        self.rz_gate = tq.RZ(has_params=True, trainable=True)
        self.measure = tq.MeasureAll(tq.PauliZ)

    @tq.static_support
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: input tensor of shape (batch, 4) with values in [-π, π]
        :return: tensor of shape (batch, 4) containing expectation values of Z on each qubit
        """
        self.q_device.reset_states(x.shape[0])
        for i in range(self.n_wires):
            self.rx_gate(self.q_device, wires=i, params=x[:, i])
        self.var_layer(self.q_device)
        self.ry_gate(self.q_device, wires=1)
        self.rz_gate(self.q_device, wires=3)
        return self.measure(self.q_device)

__all__ = ["CombinedEstimatorQNN"]
