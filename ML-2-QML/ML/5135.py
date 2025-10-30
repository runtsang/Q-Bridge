import torch
from torch import nn
import torch.nn.functional as F

# Import the seed modules
from QCNN import QCNNModel
from FraudDetection import build_fraud_detection_program, FraudLayerParameters
from SamplerQNN import SamplerQNN

class HybridQCNNFraudSampler(nn.Module):
    """
    Hybrid model that chains a classical QCNN, a photonic fraud‑detection
    inspired linear network, and a probabilistic sampler.
    """
    def __init__(self,
                 fraud_input: FraudLayerParameters,
                 fraud_layers: list[FraudLayerParameters]):
        super().__init__()
        # Classical QCNN feature extractor
        self.qcnn = QCNNModel()
        # Classical fraud‑detection network mirroring the photonic program
        self.fraud_net = build_fraud_detection_program(fraud_input, fraud_layers)
        # Probabilistic sampler (softmax output)
        self.sampler = SamplerQNN()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. QCNN feature extraction
        features = self.qcnn(x)
        # 2. Fraud detection transformation
        fraud_out = self.fraud_net(features)
        # 3. Generate probability distribution
        probs = self.sampler(fraud_out)
        return probs

__all__ = ["HybridQCNNFraudSampler", "FraudLayerParameters"]
