import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
from qiskit.circuit import ParameterVector, QuantumCircuit
from qiskit.primitives import StatevectorSampler

class QLSTM(nn.Module):
    """
    Quantum LSTM cell using torchquantum operations.
    """
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int) -> None:
            super().__init__()
            self.encoder = tq.GeneralEncoder(
                [
                    {"input_idx": [0], "func": "rx", "wires": [0]},
                    {"input_idx": [1], "func": "rx", "wires": [1]},
                    {"input_idx": [2], "func": "rx", "wires": [2]},
                    {"input_idx": [3], "func": "rx", "wires": [3]},
                ]
            )
            self.params = nn.ModuleList([tq.RX(has_params=True, trainable=True) for _ in range(n_wires)])
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=x.shape[0], device=x.device)
            self.encoder(qdev, x)
            for wire, gate in enumerate(self.params):
                gate(qdev, wires=wire)
            for wire in range(self.n_wires):
                if wire == self.n_wires - 1:
                    tqf.cnot(qdev, wires=[wire, 0])
                else:
                    tqf.cnot(qdev, wires=[wire, wire + 1])
            return self.measure(qdev)

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        self.forget = self.QLayer(n_qubits)
        self.input = self.QLayer(n_qubits)
        self.update = self.QLayer(n_qubits)
        self.output = self.QLayer(n_qubits)

        self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)

    def forward(self, inputs: torch.Tensor, states: tuple | None = None):
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget(self.linear_forget(combined)))
            i = torch.sigmoid(self.input(self.linear_input(combined)))
            g = torch.tanh(self.update(self.linear_update(combined)))
            o = torch.sigmoid(self.output(self.linear_output(combined)))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        outputs = torch.cat(outputs, dim=0)
        return outputs, (hx, cx)

    def _init_states(self, inputs: torch.Tensor, states: tuple | None = None):
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (
            torch.zeros(batch_size, self.hidden_dim, device=device),
            torch.zeros(batch_size, self.hidden_dim, device=device),
        )

class SamplerQLSTM(nn.Module):
    """
    Quantum hybrid sampler: QLSTM encoder + quantum sampler head.
    """
    def __init__(self, input_dim: int = 2, hidden_dim: int = 4, n_qubits: int = 2):
        super().__init__()
        self.lstm = QLSTM(input_dim, hidden_dim, n_qubits)
        self.inputs2 = ParameterVector("input", 2)
        self.weights2 = ParameterVector("weight", 4)
        self.qc2 = QuantumCircuit(6)  # 2 input + 4 weight qubits
        self.qc2.ry(self.inputs2[0], 0)
        self.qc2.ry(self.inputs2[1], 1)
        self.qc2.cx(0, 1)
        self.qc2.ry(self.weights2[0], 0)
        self.qc2.ry(self.weights2[1], 1)
        self.qc2.cx(0, 1)
        self.qc2.ry(self.weights2[2], 0)
        self.qc2.ry(self.weights2[3], 1)
        self.sampler = StatevectorSampler()
        self.weight_params = nn.Parameter(torch.randn(4))

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(seq)
        final_hidden = lstm_out[:, -1, :]  # (batch, hidden_dim)
        probs_list = []
        for i in range(final_hidden.size(0)):
            hidden = final_hidden[i]
            param_dict = {
                self.inputs2[0]: hidden[0].item(),
                self.inputs2[1]: hidden[1].item(),
                self.weights2[0]: self.weight_params[0].item(),
                self.weights2[1]: self.weight_params[1].item(),
                self.weights2[2]: self.weight_params[2].item(),
                self.weights2[3]: self.weight_params[3].item(),
            }
            circ = self.qc2.assign_parameters(param_dict, inplace=False)
            result = self.sampler.run(circ).result()
            statevector = result.statevector
            probs_dict = statevector.probabilities_dict()
            p0 = probs_dict.get('00', 0.0) + probs_dict.get('10', 0.0)
            p1 = probs_dict.get('01', 0.0) + probs_dict.get('11', 0.0)
            probs_tensor = torch.tensor([p0, p1], device=seq.device)
            probs_list.append(probs_tensor)
        probs = torch.stack(probs_list)
        probs = F.softmax(probs, dim=-1)
        return probs

def SamplerQNN() -> SamplerQLSTM:
    """
    Helper to match the original anchor API.
    """
    return SamplerQLSTM()

__all__ = ["SamplerQLSTM", "SamplerQNN"]
