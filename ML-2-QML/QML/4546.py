import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf
import networkx as nx
import itertools
import numpy as np

class SamplerQNN(nn.Module):
    """
    Quantum sampler that extends the original SamplerQNN.
    Combines:
    - Parameterized quantum circuit (Ry + CX) to produce a quantum state,
    - Quantum regression head using a linear readout,
    - Optional quantum LSTM cell for sequence processing,
    - Graph utilities to build fidelity adjacency from sampled states.
    """

    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "ry", "wires": [i]} for i in range(n_wires)]
            )
            self.params = nn.ModuleList([tq.RY(has_params=True, trainable=True) for _ in range(n_wires)])
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor):
            qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=x.shape[0], device=x.device)
            self.encoder(qdev, x)
            for wire, gate in enumerate(self.params):
                gate(qdev, wires=wire)
            for wire in range(self.n_wires - 1):
                tqf.cnot(qdev, wires=[wire, wire + 1])
            tqf.cnot(qdev, wires=[self.n_wires - 1, 0])
            return self.measure(qdev)

    class QLSTM(nn.Module):
        """
        Quantum LSTM cell where gates are realised by small quantum circuits.
        """
        class GateLayer(tq.QuantumModule):
            def __init__(self, n_wires: int):
                super().__init__()
                self.n_wires = n_wires
                self.encoder = tq.GeneralEncoder(
                    [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
                )
                self.params = nn.ModuleList([tq.RX(has_params=True, trainable=True) for _ in range(n_wires)])
                self.measure = tq.MeasureAll(tq.PauliZ)

            def forward(self, x: torch.Tensor):
                qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=x.shape[0], device=x.device)
                self.encoder(qdev, x)
                for wire, gate in enumerate(self.params):
                    gate(qdev, wires=wire)
                for wire in range(self.n_wires - 1):
                    tqf.cnot(qdev, wires=[wire, wire + 1])
                tqf.cnot(qdev, wires=[self.n_wires - 1, 0])
                return self.measure(qdev)

        def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int):
            super().__init__()
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.n_qubits = n_qubits
            self.forget = self.GateLayer(n_qubits)
            self.input = self.GateLayer(n_qubits)
            self.update = self.GateLayer(n_qubits)
            self.output = self.GateLayer(n_qubits)
            self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
            self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
            self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
            self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)

        def forward(self, inputs: torch.Tensor, states: tuple = None):
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

        def _init_states(self, inputs: torch.Tensor, states: tuple = None):
            if states is not None:
                return states
            batch_size = inputs.size(1)
            device = inputs.device
            return (torch.zeros(batch_size, self.hidden_dim, device=device),
                    torch.zeros(batch_size, self.hidden_dim, device=device))

    def __init__(self,
                 input_dim: int = 2,
                 hidden_dim: int = 4,
                 n_qubits: int = 2,
                 use_lstm: bool = False,
                 use_graph: bool = False,
                 regression_head: bool = False):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.use_lstm = use_lstm
        self.use_graph = use_graph
        self.regression_head = regression_head

        # Parameterized quantum circuit
        self.circuit_layer = self.QLayer(n_qubits)

        # Regression head
        if regression_head:
            self.reg_head = nn.Linear(2 ** n_qubits, 1)

        # Quantum LSTM
        if use_lstm:
            self.lstm = self.QLSTM(input_dim, hidden_dim, n_qubits)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: Tensor of shape (batch, input_dim) or (batch, seq_len, input_dim)
        Returns:
            dict with keys:
                'probs'      – sampling distribution probabilities,
               'regression' – optional regression output,
                'graph'      – optional adjacency graph.
        """
        if self.use_lstm:
            x, _ = self.lstm(x)

        probs = self.circuit_layer(x)

        out = {"probs": probs}

        if self.regression_head:
            reg = self.reg_head(probs)
            out["regression"] = reg.squeeze(-1)

        if self.use_graph:
            probs_np = probs.detach().cpu().numpy()
            graph = nx.Graph()
            graph.add_nodes_from(range(probs_np.shape[0]))
            for i, j in itertools.combinations(range(probs_np.shape[0]), 2):
                fid = np.dot(probs_np[i], probs_np[j]) ** 2
                if fid > 0.5:
                    graph.add_edge(i, j, weight=1.0)
            out["graph"] = graph

        return out

    def generate_data(self, num_samples: int = 1000):
        x = np.random.uniform(-1.0, 1.0, size=(num_samples, self.input_dim)).astype(np.float32)
        angles = x.sum(axis=1)
        y = np.sin(angles) + 0.1 * np.cos(2 * angles)
        return torch.tensor(x), torch.tensor(y)

__all__ = ["SamplerQNN"]
