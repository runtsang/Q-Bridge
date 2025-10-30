import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridFunction(torch.autograd.Function):
    """Differentiable sigmoid activation with optional shift, mirroring the quantum head."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, shift: float = 0.0) -> torch.Tensor:
        outputs = torch.sigmoid(inputs + shift)
        ctx.save_for_backward(outputs)
        return outputs

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (outputs,) = ctx.saved_tensors
        grad_inputs = grad_output * outputs * (1 - outputs)
        return grad_inputs, None

class HybridBinaryClassifier(nn.Module):
    """
    Classical CNN with a hybrid sigmoid head, structurally mirroring the QCNN‑style quantum model.
    The network consists of:
        * Two convolutional blocks with ReLU, max‑pooling and dropout.
        * A feature‑map inspired linear block followed by three “convolution” layers
          (implemented as linear+activation) and two pooling layers.
        * A fully‑connected head producing a single logit.
        * The HybridFunction applies a sigmoid to obtain a probability for class 1.
    """
    def __init__(self, shift: float = 0.0) -> None:
        super().__init__()
        # Convolutional front‑end
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)

        # Feature‑map and QCNN‑style layers
        self.feature_map = nn.Sequential(nn.Linear(8, 16), nn.Tanh())
        self.conv1_fc = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1_fc = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2_fc = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2_fc = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3_fc = nn.Sequential(nn.Linear(4, 4), nn.Tanh())

        # Dynamically infer the size of the convolutional flatten output
        dummy = torch.zeros(1, 3, 32, 32)  # input assumed 32×32 RGB
        x = F.relu(self.conv1(dummy))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        conv_out = torch.flatten(x, 1)
        conv_size = conv_out.shape[1]

        # Fully‑connected head
        self.fc1 = nn.Linear(conv_size + 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Convolutional branch
        x = F.relu(self.conv1(inputs))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)

        # QCNN‑style branch – expects 8‑dimensional feature vector
        qcnn_in = inputs.view(inputs.size(0), -1)[:, :8]
        y = self.feature_map(qcnn_in)
        y = self.conv1_fc(y)
        y = self.pool1_fc(y)
        y = self.conv2_fc(y)
        y = self.pool2_fc(y)
        y = self.conv3_fc(y)

        # Concatenate both branches
        x = torch.cat([x, y], dim=1)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        # Hybrid sigmoid head
        prob = HybridFunction.apply(x, self.shift)
        return torch.cat((prob, 1 - prob), dim=-1)

class QCNet(HybridBinaryClassifier):
    """Alias kept for backward compatibility with the original anchor."""
    pass

__all__ = ["HybridBinaryClassifier", "HybridFunction", "QCNet"]
