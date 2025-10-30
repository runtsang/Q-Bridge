import torch
import torch.nn as nn
import torch.nn.functional as F

class Hybrid(nn.Module):
    """
    Classical head that can be either a sigmoid or a small MLP.
    """
    def __init__(self, head_type: str = "sigmoid", mlp_hidden: int = 32):
        super().__init__()
        self.head_type = head_type
        if head_type == "mlp":
            self.head = nn.Sequential(
                nn.Linear(1, mlp_hidden),
                nn.ReLU(),
                nn.Linear(mlp_hidden, 1),
                nn.Sigmoid()
            )
        else:
            self.head = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)

class HybridBinaryClassifier(nn.Module):
    """
    Classical CNN backbone followed by a hybrid head.
    """
    def __init__(self, head_type: str = "sigmoid", mlp_hidden: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        self.hybrid = Hybrid(head_type, mlp_hidden)
        self.dropout_schedule = [0.2, 0.5]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        logits = self.hybrid(x)
        probs = torch.cat([logits, 1 - logits], dim=-1)
        return probs

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            probs = self(x)
        return probs.argmax(dim=-1)
