"""
ConvFusion: Hybrid classical‑quantum convolution filter.
"""

import torch
from torch import nn
from torch.nn.functional import sigmoid

# Import the quantum filter implementation from the QML module.
# The QML module defines a class named ConvFusion that implements a variational circuit.
# We alias it to avoid name clash.
try:
    from.qml_conv import ConvFusion as QuantumConvFusion
except Exception:
    # Fallback for standalone usage when the QML module is not available.
    QuantumConvFusion = None

class ConvFusion(nn.Module):
    """
    Hybrid convolutional layer that fuses a trainable classical convolution
    with a quantum variational circuit.  The output is the mean of the
    classical sigmoid activation and the quantum measurement probability
    for each receptive field.
    """

    def __init__(self,
                 kernel_size: int = 2,
                 conv_in_channels: int = 1,
                 conv_out_channels: int = 1,
                 conv_bias: bool = True,
                 quantum_shots: int = 1024,
                 quantum_backend: str = "qasm_simulator",
                 threshold: float = 0.0):
        super().__init__()
        # Classical convolution
        self.conv = nn.Conv2d(conv_in_channels,
                              conv_out_channels,
                              kernel_size=kernel_size,
                              bias=conv_bias)
        # Learnable threshold for classical activation
        self.threshold = nn.Parameter(torch.tensor(threshold, dtype=torch.float32))

        # Quantum filter
        if QuantumConvFusion is not None:
            self.quantum = QuantumConvFusion(kernel_size=kernel_size,
                                             backend=quantum_backend,
                                             shots=quantum_shots,
                                             threshold=threshold)
        else:
            # If QML module is missing, fall back to a dummy quantum pass
            self.quantum = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that returns a hybrid activation map.

        Args:
            x: Input tensor of shape (batch, in_channels, H, W).

        Returns:
            Tensor of shape (batch, out_channels, H', W') where each element
            is the mean of the classical sigmoid activation and the quantum
            measurement probability for the corresponding receptive field.
        """
        # Classical convolution
        conv_out = self.conv(x)
        act = sigmoid(conv_out - self.threshold)

        # Quantum part: compute average probability over receptive fields
        batch, ch, h, w = act.shape
        kernel = self.conv.kernel_size[0]

        # Extract patches
        patches = act.unfold(2, kernel, 1).unfold(3, kernel, 1)
        # patches shape: (batch, ch, h', w', kernel, kernel)
        patches = patches.contiguous().view(-1, kernel, kernel)

        # Run quantum filter on each patch
        quantum_probs = []
        for patch in patches:
            patch_np = patch.detach().cpu().numpy()
            prob = self.quantum.run(patch_np) if self.quantum is not None else 0.0
            quantum_probs.append(prob)
        quantum_probs = torch.tensor(quantum_probs, device=act.device)

        # Reshape back to (batch, ch, h', w')
        quantum_probs = quantum_probs.view(batch, ch, h - kernel + 1, w - kernel + 1)

        # Combine classical and quantum activations by averaging
        return (act + quantum_probs) / 2.0
