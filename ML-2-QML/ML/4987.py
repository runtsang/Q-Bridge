import torch
import torch.nn as nn
import torch.nn.functional as F

# Import the quantum components defined in the QML block.
# They are expected to be available in the same package/module.
# The QML block defines `SamplerQNN` (a qiskit SamplerQNN instance)
# and `QuanvolutionFilter` (a torchquantum QuantumModule).
try:
    from. import SamplerQNN as QSamplerQNN, QuanvolutionFilter as QQuanvolutionFilter
except ImportError:
    # Fallback for environments where the QML module is not yet imported.
    # The user must ensure that the QML definitions are available.
    QSamplerQNN = None
    QQuanvolutionFilter = None

class QuantumSamplerAdapter(nn.Module):
    """
    Adapts a qiskit SamplerQNN to behave like a PyTorch module.
    The underlying qiskit circuit is executed on the CPU and the resulting
    probability distribution is converted back to a torch.Tensor.
    """
    def __init__(self, sampler: QSamplerQNN, input_dim: int = 2):
        super().__init__()
        if sampler is None:
            raise RuntimeError("Quantum sampler must be provided.")
        self.sampler = sampler
        self.input_dim = input_dim

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Ensure inputs are on CPU for qiskit execution
        inp_np = inputs.detach().cpu().numpy()
        # qiskit SamplerQNN returns a dict; we extract the 'values' key
        probs = self.sampler(inp_np).values
        return torch.from_numpy(probs).to(inputs.device)

class HybridSamplerRegressor(nn.Module):
    """
    Hybrid model that combines:
      * a quantum sampler (or quanvolution) for feature extraction,
      * a classical linear head for regression.
    """
    def __init__(self, input_dim: int = 2, use_quanvolution: bool = False):
        super().__init__()
        self.use_quanvolution = use_quanvolution

        if self.use_quanvolution:
            if QQuanvolutionFilter is None:
                raise RuntimeError("QuanvolutionFilter is not available.")
            self.quanvolution = QQuanvolutionFilter()
            # The output of QuanvolutionFilter is (batch, 4*14*14)
            self.head = nn.Linear(4 * 14 * 14, 1)
        else:
            if QSamplerQNN is None:
                raise RuntimeError("Quantum SamplerQNN is not available.")
            quantum_sampler = QSamplerQNN()
            self.quantum_sampler = QuantumSamplerAdapter(quantum_sampler, input_dim=input_dim)
            self.head = nn.Linear(2, 1)  # SamplerQNN outputs 2 probabilities

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            For sampler mode: shape (batch, input_dim)
            For quanvolution mode: shape (batch, 1, 28, 28)
        Returns
        -------
        torch.Tensor
            Regression output of shape (batch,)
        """
        if self.use_quanvolution:
            features = self.quanvolution(x)
            logits = self.head(features)
        else:
            probs = self.quantum_sampler(x)
            logits = self.head(probs)
        return logits.squeeze(-1)

__all__ = ["HybridSamplerRegressor", "QuantumSamplerAdapter"]
