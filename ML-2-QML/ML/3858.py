import torch
from torch import nn
from.QuantumKernelMethod import Kernel as ClassicalKernel

class EstimatorQNNHybrid(nn.Module):
    """
    Hybrid estimator supporting both classical RBF and quantum kernels.
    Training is performed via kernel ridge regression.
    """
    def __init__(self, kernel_type: str = "quantum", gamma: float = 1.0,
                 n_wires: int = 4, lambda_reg: float = 1e-3):
        super().__init__()
        self.kernel_type = kernel_type
        self.lambda_reg = lambda_reg
        if kernel_type == "quantum":
            from.QuantumKernelMethod import Kernel as QuantumKernel
            self.kernel = QuantumKernel()
        else:
            self.kernel = ClassicalKernel(gamma)
        self.w = None
        self.train_X = None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.train_X is None or self.w is None:
            raise RuntimeError("Model has not been trained. Call `train` first.")
        phi = self.kernel(self.train_X, inputs).squeeze()
        return phi @ self.w

    def train(self, X: torch.Tensor, y: torch.Tensor) -> None:
        """
        Fit the model using kernel ridge regression.
        """
        K = self.kernel(X, X).squeeze()
        self.w = torch.linalg.solve(K + self.lambda_reg * torch.eye(K.shape[0]), y)
        self.train_X = X

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        return self.forward(X)

def EstimatorQNN():
    """
    Returns a default instance of the hybrid estimator using the quantum kernel.
    """
    return EstimatorQNNHybrid(kernel_type="quantum")
