import numpy as np
import torch
from torch import nn

class HybridSelfAttentionCNN:
    """
    Classical hybrid model that combines a self‑attention block with a small CNN
    stack (convolution + pooling) inspired by the QCNN implementation.
    The interface mirrors the original SelfAttention and QCNN seeds:
        run(rotation_params, entangle_params, inputs)
    """

    def __init__(self, embed_dim: int = 4, input_dim: int = 8) -> None:
        self.embed_dim = embed_dim
        self.input_dim = input_dim

        # Self‑attention weight matrices are expected to be provided at run‑time.
        # They are not part of the model parameters, just placeholders for the
        # interface compatibility.
        # Convolution‑pooling backbone (fixed architecture)
        self.feature_map = nn.Sequential(nn.Linear(input_dim, 8), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(8, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
        self.head = nn.Linear(4, 1)

    def _attention(self,
                   rotation_params: np.ndarray,
                   entangle_params: np.ndarray,
                   inputs: torch.Tensor) -> torch.Tensor:
        """
        Compute a scaled‑dot‑product self‑attention for the input batch.
        Parameters are supplied as 1‑D numpy arrays and reshaped on the fly.
        """
        # Prepare weight matrices
        W_q = torch.from_numpy(rotation_params.reshape(self.embed_dim, -1)).float()
        W_k = torch.from_numpy(entangle_params.reshape(self.embed_dim, -1)).float()
        # Query, key and value
        Q = torch.matmul(inputs, W_q.t())
        K = torch.matmul(inputs, W_k.t())
        V = inputs
        # Attention scores
        scores = torch.softmax(torch.matmul(Q, K.t()) / np.sqrt(self.embed_dim), dim=-1)
        return torch.matmul(scores, V)

    def forward(self,
                inputs: torch.Tensor,
                rotation_params: np.ndarray,
                entangle_params: np.ndarray) -> torch.Tensor:
        """
        Forward pass through attention + CNN stack.
        """
        x = self._attention(rotation_params, entangle_params, inputs)
        x = self.feature_map(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return torch.sigmoid(self.head(x))

    def run(self,
            rotation_params: np.ndarray,
            entangle_params: np.ndarray,
            inputs: np.ndarray) -> np.ndarray:
        """
        Convenience method that accepts raw numpy arrays and returns a numpy
        array of predictions.
        """
        inputs_t = torch.from_numpy(inputs.astype(np.float32))
        with torch.no_grad():
            out = self.forward(inputs_t, rotation_params, entangle_params)
        return out.numpy()
