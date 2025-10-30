import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf


def EstimatorQNN():
    """
    Lightweight feed‑forward network that mimics the quantum
    EstimatorQNN example.  It outputs a single scalar per batch.
    """
    class EstimatorNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(2, 8),
                nn.Tanh(),
                nn.Linear(8, 4),
                nn.Tanh(),
                nn.Linear(4, 1),
            )

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return self.net(inputs)

    return EstimatorNN()


class HybridQLSTM(nn.Module):
    """
    Classical‑quantum hybrid LSTM.
    * Classical linear gates when `n_qubits==0`.
    * Quantum gates realised by a small variational circuit
      (`QLayer`) when `n_qubits>0`.
    * Optional `EstimatorQNN` scalar multiplier for each gate.
    """
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
            )
            self.params = nn.ModuleList([tq.RX(has_params=True, trainable=True) for _ in range(n_wires)])
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=x.shape[0], device=x.device)
            self.encoder(qdev, x)
            for wire, gate in enumerate(self.params):
                gate(qdev, wires=wire)
            # simple entangling CNOT chain
            for wire in range(self.n_wires - 1):
                tqf.cnot(qdev, wires=[wire, wire + 1])
            return self.measure(qdev)

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int = 0, use_estimator: bool = False):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.use_estimator = use_estimator

        if n_qubits > 0:
            if hidden_dim!= n_qubits:
                raise ValueError("For quantum mode, hidden_dim must equal n_qubits.")
            self.forget = self.QLayer(n_qubits)
            self.input = self.QLayer(n_qubits)
            self.update = self.QLayer(n_qubits)
            self.output = self.QLayer(n_qubits)
            self.projection = nn.Linear(input_dim + hidden_dim, n_qubits, bias=False)
        else:
            self.forget_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
            self.input_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
            self.update_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
            self.output_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)

        self.estimator = EstimatorQNN() if use_estimator else None

    def forward(
        self,
        inputs: torch.Tensor,
        states: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []

        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)

            if self.n_qubits > 0:
                proj = self.projection(combined)
                f = torch.sigmoid(self.forget(proj))
                i = torch.sigmoid(self.input(proj))
                g = torch.tanh(self.update(proj))
                o = torch.sigmoid(self.output(proj))
            else:
                f = torch.sigmoid(self.forget_linear(combined))
                i = torch.sigmoid(self.input_linear(combined))
                g = torch.tanh(self.update_linear(combined))
                o = torch.sigmoid(self.output_linear(combined))

            if self.estimator is not None:
                # Estimator expects 2‑dim input; slice if necessary
                scale = self.estimator(combined[:, :2]) if combined.shape[1] >= 2 else torch.zeros(
                    combined.shape[0], 1, device=combined.device
                )
                f, i, g, o = f * scale, i * scale, g * scale, o * scale

            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))

        stacked = torch.cat(outputs, dim=0)
        return stacked, (hx, cx)

    @staticmethod
    def _init_states(
        inputs: torch.Tensor,
        states: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        hx = torch.zeros(batch_size, inputs.size(0), device=device)
        cx = torch.zeros(batch_size, inputs.size(0), device=device)
        return hx, cx


__all__ = ["HybridQLSTM"]
