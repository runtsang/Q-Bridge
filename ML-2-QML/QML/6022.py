"""UnifiedSamplerQLSTM: Quantum‑enhanced sampler and LSTM.

The module builds on the QLSTM implementation that uses torchquantum
for the LSTM gates and a qiskit‑based SamplerQNN for the sampler.
Both components are fully differentiable and can be trained end‑to‑end.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

# --- Quantum Sampler -----------------------------------------
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit.primitives import StatevectorSampler

class SamplerModuleQuantum(nn.Module):
    """
    Parameterized quantum sampler implemented with qiskit.

    Parameters
    ----------
    input_dim : int
        Number of input parameters (qubits to encode the data).
    hidden_dim : int
        Size of the hidden layer in the classical encoder.
    output_dim : int
        Dimension of the sampler output (number of weight parameters).
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.input_params = ParameterVector("input", input_dim)
        self.weight_params = ParameterVector("weight", output_dim)
        qc = QuantumCircuit(input_dim)
        # Encode data with Ry rotations
        for i in range(input_dim):
            qc.ry(self.input_params[i], i)
        # Entangling layer
        for i in range(input_dim - 1):
            qc.cx(i, i + 1)
        # Apply weight rotations on the same qubits
        for i in range(output_dim):
            qc.ry(self.weight_params[i], i % input_dim)
        # Final entanglement
        for i in range(input_dim - 1):
            qc.cx(i, i + 1)
        sampler = StatevectorSampler()
        self.sampler_qnn = SamplerQNN(
            circuit=qc,
            input_params=self.input_params,
            weight_params=self.weight_params,
            sampler=sampler,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs shape: (batch, input_dim)
        return self.sampler_qnn(inputs)

# --- Quantum LSTM --------------------------------------------
import torchquantum as tq
import torchquantum.functional as tqf

class QLSTM(nn.Module):
    """
    Quantum LSTM cell where each gate is a small quantum circuit.
    """
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int) -> None:
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

    def forward(self, inputs: torch.Tensor, states: Tuple[torch.Tensor, torch.Tensor] | None = None):
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

    def _init_states(
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (
            torch.zeros(batch_size, self.hidden_dim, device=device),
            torch.zeros(batch_size, self.hidden_dim, device=device),
        )

class LSTMTagger(nn.Module):
    """
    Sequence tagging model that can switch between classical and quantum LSTM.
    """
    def __init__(self, input_dim: int, hidden_dim: int, tagset_size: int, n_qubits: int = 0) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        if n_qubits > 0:
            self.lstm = QLSTM(input_dim, hidden_dim, n_qubits=n_qubits)
        else:
            self.lstm = nn.LSTM(input_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(inputs.permute(1,0,2))
        tag_logits = self.hidden2tag(lstm_out.permute(1,0,2))
        return F.log_softmax(tag_logits, dim=2)

class UnifiedSamplerQLSTM(nn.Module):
    """
    Hybrid sampler‑LSTM module with quantum gates.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the raw feature vector fed to the sampler.
    hidden_dim : int
        Dimensionality of the sampler output and the hidden state of the LSTM.
    tagset_size : int
        Number of target tags.
    n_qubits : int, optional
        If >0, quantum LSTM gates are used; otherwise classical LSTM.
    """
    def __init__(self, input_dim: int, hidden_dim: int, tagset_size: int, n_qubits: int = 0):
        super().__init__()
        self.sampler = SamplerModuleQuantum(input_dim, hidden_dim, output_dim=hidden_dim)
        self.tagger = LSTMTagger(hidden_dim, hidden_dim, tagset_size, n_qubits=n_qubits)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs shape: (batch, seq_len, input_dim)
        batch, seq_len, _ = inputs.shape
        flat_inputs = inputs.view(batch*seq_len, -1)
        probs = self.sampler(flat_inputs)
        sampled = probs.view(batch, seq_len, -1)
        return self.tagger(sampled)

__all__ = ["UnifiedSamplerQLSTM"]
