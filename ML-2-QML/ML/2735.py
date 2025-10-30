import torch
import torch.nn as nn
import torch.nn.functional as F

class QCNNModel(nn.Module):
    """Classical emulation of a QCNN feature extractor."""
    def __init__(self):
        super().__init__()
        self.feature_map = nn.Sequential(nn.Linear(8, 16), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
        self.head = nn.Linear(4, 1)

    def forward(self, x):
        x = self.feature_map(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return torch.sigmoid(self.head(x))

class HybridQLSTM(nn.Module):
    """Classical LSTM with optional QCNN feature extractor."""
    def __init__(self, input_dim, hidden_dim, tagset_size, use_qcnn=False):
        super().__init__()
        self.use_qcnn = use_qcnn
        self.feature_extractor = QCNNModel() if use_qcnn else None
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, inputs):
        if self.use_qcnn:
            batch, seq, feat = inputs.shape
            flat = inputs.view(batch * seq, feat)
            features = self.feature_extractor(flat)
            inputs = features.view(batch, seq, -1)
        outputs, _ = self.lstm(inputs)
        logits = self.hidden2tag(outputs)
        return F.log_softmax(logits, dim=-1)

# Alias for backward compatibility
QLSTM = HybridQLSTM

__all__ = ["HybridQLSTM", "QCNNModel", "QLSTM"]
