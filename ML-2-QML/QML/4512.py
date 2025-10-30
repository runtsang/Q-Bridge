import torch
import torch.nn as nn
import torchquantum as tq
import networkx as nx
import numpy as np
from torchquantum.functional import func_name_dict


class KernalAnsatz(tq.QuantumModule):
    """Programmable quantum circuit for the kernel."""
    def __init__(self, func_list):
        super().__init__()
        self.func_list = func_list

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        q_device.reset_states(x.shape[0])
        for info in self.func_list:
            params = x[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)
        for info in reversed(self.func_list):
            params = -y[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)


class Kernel(tq.QuantumModule):
    """Quantum kernel evaluated via a fixed ansatz."""
    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.ansatz = KernalAnsatz(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.ansatz(self.q_device, x, y)
        return torch.abs(self.q_device.states.view(-1)[0])


class QuantumGraphNetwork(tq.QuantumModule):
    """Parameterized quantum network that propagates information over a fixed graph."""
    def __init__(self, adjacency: nx.Graph):
        super().__init__()
        self.adj = adjacency
        self.n_wires = adjacency.number_of_nodes()
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.params = nn.Parameter(torch.randn(self.n_wires))
        self.edges = list(adjacency.edges())

    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor) -> torch.Tensor:
        q_device.reset_states(x.shape[0])
        for i in range(self.n_wires):
            tq.ry(q_device, wires=[i], params=x[:, i:i+1])
        for i in range(self.n_wires):
            tq.ry(q_device, wires=[i], params=self.params[i])
        for i, j in self.edges:
            tq.cz(q_device, wires=[i, j])
        return q_device.states.view(-1).real


class QuanvolutionFilter(tq.QuantumModule):
    """Quantum convolution that processes 2x2 patches of a grayscale image."""
    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        self.q_layer = tq.RandomLayer(n_ops=8, wires=list(range(self.n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        device = x.device
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)
        x = x.view(bsz, 28, 28)
        patches = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                data = torch.stack(
                    [
                        x[:, r, c],
                        x[:, r, c + 1],
                        x[:, r + 1, c],
                        x[:, r + 1, c + 1],
                    ],
                    dim=1,
                )
                self.encoder(qdev, data)
                self.q_layer(qdev)
                measurement = self.measure(qdev)
                patches.append(measurement.view(bsz, 4))
        return torch.cat(patches, dim=1)


class QuanvolutionGen196(tq.QuantumModule):
    """Quantum‑enhanced network that mirrors the classical hybrid architecture."""
    def __init__(
        self,
        prototype_vectors: torch.Tensor,
        kernel_threshold: float = 0.8,
        graph_arch: list[int] = [4, 4, 10]
    ):
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        self.kernel = Kernel()
        self.prototypes = prototype_vectors  # shape (P, 4)

        # Build prototype adjacency offline
        prot_mat = self.prototypes.unsqueeze(0)
        prot_mat_t = self.prototypes.unsqueeze(1)
        sim = torch.exp(-torch.sum((prot_mat - prot_mat_t) ** 2, dim=-1))
        adj = (sim >= kernel_threshold).float()
        self.graph_network = QuantumGraphNetwork(nx.from_numpy_array(adj.numpy()))
        self.final = nn.Identity()  # output dimension matches graph network

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Quantum convolution
        patch_meas = self.qfilter(x)                    # B x (P*4)
        B, P4 = patch_meas.shape
        P = P4 // 4
        patch_meas = patch_meas.view(B, P, 4)

        # 2. Quantum kernel similarity to prototypes
        k_mat = torch.zeros(B, P, self.prototypes.size(0), device=x.device)
        for i in range(B):
            for j in range(P):
                for k in range(self.prototypes.size(0)):
                    k_mat[i, j, k] = self.kernel(patch_meas[i, j], self.prototypes[k])

        # 3. Aggregate prototype‑weighted features
        agg = torch.sum(k_mat.unsqueeze(-1) * self.prototypes, dim=1)  # B x P x 4
        agg_flat = agg.view(B, -1)                                   # B x (P*4)

        # 4. Graph‑based quantum propagation
        graph_out = self.graph_network(self.graph_network.q_device, agg_flat)

        logits = self.final(graph_out)
        return F.log_softmax(logits, dim=-1)


__all__ = ["KernalAnsatz", "Kernel", "QuanvolutionFilter", "QuantumGraphNetwork", "QuanvolutionGen196"]
