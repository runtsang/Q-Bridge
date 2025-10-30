import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """Adds a residual connection to the linear feature extractor."""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.linear(x)
        out = self.bn(out)
        out = self.relu(out)
        return out + x

class ClassicalHead(nn.Module):
    """Classical sigmoid head with learnable bias and scaling."""
    def __init__(self, in_features, bias=True, scale=True):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        if bias:
            self.bias = nn.Parameter(torch.zeros(1))
        else:
            self.bias = None
        if scale:
            self.amp = nn.Parameter(torch.tensor([1.0]))
        else:
            self.amp = None

    def forward(self, x):
        logits = self.linear(x)
        if self.bias is not None:
            logits = logits + self.bias
        if self.amp is not None:
            logits = logits * self.amp
        probs = torch.sigmoid(logits)
        return torch.cat((probs, 1 - probs), dim=-1)

class QuantumHybridClassifier(nn.Module):
    """Classical CNN followed by a residual block and a calibrated sigmoid head."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.residual = ResidualBlock(120, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        self.head = ClassicalHead(1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.residual(x)
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return self.head(x)
