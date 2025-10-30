import torch  
import torch.nn as nn  
import torch.nn.functional as F  

class QuanvolutionNet(nn.Module):  
    """Classical analogue of the Quanvolution model.  
    Replaces the quantum filter with a learnable 2×2 convolutional filter  
    followed by a depthwise convolution to emulate entanglement.  
    The linear head remains unchanged."""  

    def __init__(self):  
        super().__init__()  
        # 2×2 convolution with stride 2, output channels 4  
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2, bias=False)  
        # Depthwise convolution to emulate inter‑patch coupling  
        self.depthwise = nn.Conv2d(4, 4, kernel_size=1, groups=4, bias=False)  
        self.linear = nn.Linear(4 * 14 * 14, 10)  

    def forward(self, x: torch.Tensor) -> torch.Tensor:  
        # x: (batch, 1, 28, 28)  
        features = self.conv(x)          # (batch, 4, 14, 14)  
        features = self.depthwise(features)  # (batch, 4, 14, 14)  
        features = features.view(x.size(0), -1)  # flatten  
        logits = self.linear(features)  
        return F.log_softmax(logits, dim=-1)  

__all__ = ["QuanvolutionNet"]
