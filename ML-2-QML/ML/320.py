import torch
import torch.nn as nn
import torch.nn.functional as F

class QuanvolutionFilter(nn.Module):
    """Depth‑wise separable convolution inspired by the original quanvolution filter."""
    def __init__(self, in_channels=1, out_channels=4, kernel_size=2, stride=2, depthwise=True):
        super().__init__()
        self.depthwise = depthwise
        if depthwise:
            self.depthwise_conv = nn.Conv2d(
                in_channels, in_channels, kernel_size=kernel_size,
                stride=stride, padding=0, bias=False, groups=in_channels
            )
            self.pointwise_conv = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, bias=False
            )
        else:
            self.conv = nn.Conv2d(
                in_channels, out_channels, kernel_size=kernel_size,
                stride=stride, padding=0, bias=False
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.depthwise:
            x = self.depthwise_conv(x)
            x = self.pointwise_conv(x)
        else:
            x = self.conv(x)
        return x.view(x.size(0), -1)

class QuanvolutionClassifier(nn.Module):
    """Hybrid network using the classical quanvolutional filter followed by a linear head."""
    def __init__(self, in_channels=1, num_classes=10, depthwise=True):
        super().__init__()
        self.qfilter = QuanvolutionFilter(
            in_channels=in_channels,
            out_channels=4,
            kernel_size=2,
            stride=2,
            depthwise=depthwise
        )
        self.linear = nn.Linear(4 * 14 * 14, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)

    def fit(self, dataloader, epochs=5, lr=1e-3, device="cpu"):
        self.to(device)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for xb, yb in dataloader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                logp = self.forward(xb)
                loss = F.nll_loss(logp, yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * xb.size(0)
            avg_loss = total_loss / len(dataloader.dataset)
            print(f"Epoch {epoch+1}/{epochs} - loss: {avg_loss:.4f}")

    def evaluate(self, dataloader, device="cpu"):
        self.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for xb, yb in dataloader:
                xb, yb = xb.to(device), yb.to(device)
                logp = self.forward(xb)
                preds = logp.argmax(dim=-1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)
        return correct / total

__all__ = ["QuanvolutionFilter", "QuanvolutionClassifier"]
