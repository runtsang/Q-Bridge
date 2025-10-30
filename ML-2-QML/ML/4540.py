import torch
import torch.nn as nn
import torch.nn.functional as F

# Import quantum sub‑modules defined in the QML counterpart.
# If qml.py lives in the same package, use relative import; otherwise use absolute.
try:
    from.qml import QuanvolutionFilterQuantum, QuantumLSTM
except Exception:
    from qml import QuanvolutionFilterQuantum, QuantumLSTM


class QuanvolutionFilter(nn.Module):
    """
    Classical front‑end that slices a single‑channel image into 2×2 patches
    and forwards each patch to a small quantum kernel.
    """
    def __init__(self, n_wires: int = 4, n_ops: int = 8):
        super().__init__()
        # Conv layer only slices patches – no learnable weights.
        self.patch_conv = nn.Conv2d(1, 1, kernel_size=2, stride=2, bias=False)
        self.qfilter = QuanvolutionFilterQuantum(n_wires=n_wires, n_ops=n_ops)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: Tensor of shape (B, 1, H, W)
        :return: Tensor of shape (B, 4 * num_patches)
        """
        patches = self.patch_conv(x)  # (B, 1, H/2, W/2)
        patches = patches.view(x.size(0), -1, 4)  # (B, num_patches, 4)
        out = []
        for i in range(patches.size(1)):
            out.append(self.qfilter(patches[:, i, :]))
        return torch.cat(out, dim=1)


class QuanvolutionClassifier(nn.Module):
    """
    Hybrid classifier that uses the quantum filter followed by a linear head.
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        self.linear = nn.Linear(4 * 14 * 14, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)


class QuanvolutionGen(nn.Module):
    """
    End‑to‑end hybrid model that handles both image and sequence data.
    Image data is processed by the QuanvolutionClassifier.
    Sequence data is processed by a quantum‑augmented LSTM.
    """
    def __init__(self,
                 num_classes: int = 10,
                 lstm_hidden: int = 32,
                 lstm_layers: int = 1):
        super().__init__()
        self.image_classifier = QuanvolutionClassifier(num_classes=num_classes)
        self.lstm = QuantumLSTM(n_input=128, n_hidden=lstm_hidden, n_layers=lstm_layers)
        self.fc = nn.Linear(lstm_hidden, num_classes)

    def forward_image(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for image data.
        """
        return self.image_classifier(x)

    def forward_sequence(self, seq: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for sequential data.
        :param seq: Tensor of shape (T, B, feature_dim)
        :return: Log‑softmax over the last hidden state.
        """
        out, _ = self.lstm(seq)
        logits = self.fc(out[:, -1, :])  # last hidden state
        return F.log_softmax(logits, dim=-1)
