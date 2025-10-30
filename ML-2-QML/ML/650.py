import torch
from torch import nn
import numpy as np

class ConvEnhanced(nn.Module):
    """Classical depthwise‑separable convolution filter that can optionally initialise
    its weights from a quantum circuit.  The public interface is identical to the
    original Conv() factory: an instance exposes a run(data) method that accepts
    a 2‑D array of shape (kernel_size, kernel_size) and returns a single float
    representing the filter response.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        use_quantum: bool = False,
        threshold: float = 0.5,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.use_quantum = use_quantum
        self.device = device

        # Depthwise convolution (single channel)
        self.depthwise = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=kernel_size,
            stride=1,
            padding=0,
            bias=False,
            device=device,
        )
        # Point‑wise linear projection to scalar
        self.pointwise = nn.Linear(
            in_features=kernel_size * kernel_size,
            out_features=1,
            bias=False,
            device=device,
        )

        if self.use_quantum:
            # Attempt to initialise weights from a quantum circuit.
            try:
                from.qml import ConvEnhanced as QuantumConvEnhanced
                q_obj = QuantumConvEnhanced(kernel_size=self.kernel_size,
                                            threshold=self.threshold,
                                            device=self.device)
                kernel_matrix = q_obj.get_kernel()
                kernel_tensor = torch.tensor(kernel_matrix,
                                             dtype=torch.float32,
                                             device=self.device)
                with torch.no_grad():
                    self.depthwise.weight.copy_(kernel_tensor.unsqueeze(0).unsqueeze(0))
            except Exception:
                # Fallback: random initialization
                nn.init.xavier_uniform_(self.depthwise.weight)

    def run(self, data: np.ndarray) -> float:
        """
        Run the filter on a single patch.

        Parameters
        ----------
        data : np.ndarray
            2‑D array of shape (kernel_size, kernel_size).

        Returns
        -------
        float
            Scalar filter response.
        """
        tensor = torch.as_tensor(data, dtype=torch.float32, device=self.device)
        tensor = tensor.view(1, 1, self.kernel_size, self.kernel_size)
        logits = self.depthwise(tensor)
        activations = torch.sigmoid(logits - self.threshold)
        flat = activations.view(1, -1)
        out = self.pointwise(flat)
        return out.mean().item()

    def get_kernel(self) -> np.ndarray:
        """
        Return the current depthwise kernel as a NumPy array.
        """
        return self.depthwise.weight.detach().cpu().numpy().reshape(
            self.kernel_size, self.kernel_size
        )
