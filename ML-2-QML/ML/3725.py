import torch
from torch import nn
import torch.nn.functional as F

class UnifiedQCNN(nn.Module):
    """
    Classical convolution‑style network that extends the original QCNN seed. 
    Features a feature extractor, successive convolution and pooling layers, 
    and a dual head that can output a classification probability and, 
    optionally, a regression value.
    """
    def __init__(self,
                 input_dim: int = 8,
                 hidden_dims: list[int] | None = None,
                 num_classes: int = 1,
                 regression: bool = False):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [16, 16, 12, 8, 4, 4]
        self.feature_map = nn.Linear(input_dim, hidden_dims[0])
        self.conv1 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.pool1 = nn.Linear(hidden_dims[1], hidden_dims[2])
        self.conv2 = nn.Linear(hidden_dims[2], hidden_dims[3])
        self.pool2 = nn.Linear(hidden_dims[3], hidden_dims[4])
        self.conv3 = nn.Linear(hidden_dims[4], hidden_dims[5])
        self.tanh = nn.Tanh()
        self.classifier = nn.Linear(hidden_dims[5], num_classes)
        self.regression = regression
        if regression:
            self.regressor = nn.Linear(hidden_dims[5], 1)
        else:
            self.regressor = None

    def forward(self, x: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        x = self.tanh(self.feature_map(x))
        x = self.tanh(self.conv1(x))
        x = self.tanh(self.pool1(x))
        x = self.tanh(self.conv2(x))
        x = self.tanh(self.pool2(x))
        x = self.tanh(self.conv3(x))
        out_cls = torch.sigmoid(self.classifier(x))
        if self.regression:
            out_reg = self.regressor(x)
            return out_cls, out_reg
        return out_cls
