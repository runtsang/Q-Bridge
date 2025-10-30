import torch
from torch import nn
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.circuit import Parameter
import numpy as np

# Optional graph utilities – imported if available
try:
    from GraphQNN import random_network
except Exception:
    random_network = None

class HybridEstimatorQNN(nn.Module):
    """
    Hybrid estimator that merges a classical feed‑forward backbone,
    an optional graph‑based feature extractor, and a small variational
    quantum circuit.  The quantum part is implemented with Qiskit
    and evaluated on a state‑vector simulator; the output is added
    to the classical prediction.  The module is fully PyTorch‑
    compatible and can be trained with standard optimizers.
    """

    def __init__(self,
                 input_dim: int = 2,
                 hidden_dim: int = 8,
                 output_dim: int = 1,
                 dropout: float = 0.1,
                 use_graph: bool = False,
                 graph_arch: list[int] | None = None):
        super().__init__()

        # Classical backbone ----------------------------------------------------
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

        # Optional graph feature extractor ---------------------------------------
        self.use_graph = use_graph
        if use_graph:
            if graph_arch is None:
                graph_arch = [input_dim, 16, 8]
            # Create a small random graph network that will be used to transform
            # the input vector into graph‑based features.
            _, weights, _, _ = random_network(graph_arch, samples=1)
            self.graph_layers = nn.ModuleList()
            for w in weights:
                layer = nn.Linear(w.shape[1], w.shape[0])
                layer.weight.data = w
                layer.bias.data.zero_()
                self.graph_layers.append(layer)
        else:
            self.graph_layers = None

        # Quantum variational layer --------------------------------------------
        self.input_param = Parameter('x')
        self.w1 = Parameter('w1')
        self.w2 = Parameter('w2')
        qc = QuantumCircuit(1)
        qc.ry(self.input_param, 0)
        qc.ry(self.w1, 0)
        qc.rz(self.w2, 0)
        self.qc = qc

        # Store the trainable weights as torch parameters
        self.weight_params = nn.Parameter(torch.randn(2))

        # Observable for expectation value
        from qiskit.quantum_info import SparsePauliOp
        self.observable = SparsePauliOp.from_list([('Y', 1)])

    # -------------------------------------------------------------------------

    def _quantum_output(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the variational circuit on a state‑vector simulator.
        The first element of the input tensor is used as the input rotation.
        """
        inp = float(x[0]) if x.ndim > 0 else float(x)
        bound = {
            self.input_param: inp,
            self.w1: float(self.weight_params[0]),
            self.w2: float(self.weight_params[1]),
        }
        sv = Statevector(self.qc.bind_parameters(bound))
        exp_val = sv.expectation_value(self.observable).real
        return torch.tensor(exp_val, dtype=x.dtype, device=x.device)

    # -------------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that concatenates the classical backbone, optional
        graph features, and the quantum expectation value.
        """
        out = self.backbone(x)

        if self.use_graph and self.graph_layers is not None:
            g = x
            for layer in self.graph_layers:
                g = torch.tanh(layer(g))
            out = out + g

        q_out = self._quantum_output(x)
        out = out + q_out.unsqueeze(-1)

        return out
