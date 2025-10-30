"""Quantum hybrid network combining a quanvolution filter, fidelity graph, and a Qiskit EstimatorQNN."""  
import torch  
import torchquantum as tq  
import networkx as nx  
from EstimatorQNN import EstimatorQNN  

class QuanvolutionHybrid(tq.QuantumModule):  
    """  
    Quantum implementation of the hybrid architecture.  
    It applies a 2×2 patch encoder into qubits, aggregates the measurement  
    outcomes via a fidelity graph, and finally uses a Qiskit EstimatorQNN  
    as a quantum regressor.  
    """  

    def __init__(  
        self,  
        num_qubits: int = 4,  
        n_ops: int = 8,  
        fidelity_threshold: float = 0.8,  
    ) -> None:  
        super().__init__()  
        self.num_qubits = num_qubits  
        self.encoder = tq.GeneralEncoder(  
            [  
                {"input_idx": [0], "func": "ry", "wires": [0]},  
                {"input_idx": [1], "func": "ry", "wires": [1]},  
                {"input_idx": [2], "func": "ry", "wires": [2]},  
                {"input_idx": [3], "func": "ry", "wires": [3]},  
            ]  
        )  
        self.q_layer = tq.RandomLayer(n_ops=n_ops, wires=list(range(num_qubits)))  
        self.measure = tq.MeasureAll(tq.PauliZ)  
        self.fidelity_threshold = fidelity_threshold  
        self.estimator_qnn = EstimatorQNN()  # Qiskit EstimatorQNN instance  

    def _quantum_patches(self, x: torch.Tensor) -> torch.Tensor:  
        """Return a tensor of measurement results for each 2×2 patch."""  
        B = x.shape[0]  
        device = x.device  
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
                qdev = tq.QuantumDevice(self.num_qubits, bsz=B, device=device)  
                self.encoder(qdev, data)  
                self.q_layer(qdev)  
                measurement = self.measure(qdev)  
                patches.append(measurement.view(B, self.num_qubits))  
        return torch.cat(patches, dim=1).reshape(B, 14 * 14, self.num_qubits)  

    def _aggregate_fidelity(self, patches: torch.Tensor) -> torch.Tensor:  
        """Aggregate patch measurement vectors via a thresholded fidelity graph."""  
        B, N, F = patches.shape  
        patches_norm = patches / patches.norm(dim=2, keepdim=True).clamp_min(1e-7)  
        sims = torch.bmm(patches_norm, patches_norm.transpose(1, 2))  
        adj = (sims >= self.fidelity_threshold).float()  
        deg = adj.sum(dim=2, keepdim=True).clamp_min(1e-7)  
        agg = torch.bmm(adj, patches) / deg  
        return agg.reshape(B, -1)  

    def forward(self, x: torch.Tensor) -> torch.Tensor:  
        patches = self._quantum_patches(x)  
        agg = self._aggregate_fidelity(patches)  
        # Reduce to a scalar per sample for the EstimatorQNN  
        scalar = agg.mean(dim=1, keepdim=True)  
        # EstimatorQNN expects a batch of input parameters [input1]  
        return self.estimator_qnn(scalar)  

__all__ = ["QuanvolutionHybrid"]
