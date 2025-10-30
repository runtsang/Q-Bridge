"""ML implementation of a hybrid convolutional filter with optional classical self‑attention and auto‑encoder."""
import torch
import torch.nn as nn
import numpy as np
from Autoencoder import Autoencoder, AutoencoderConfig
from SelfAttention import SelfAttention
from QLSTM import QLSTM

class ConvGen114(nn.Module):
    """Hybrid classical convolutional filter.
    Combines a standard 2‑D convolution with optional auto‑encoding and
    self‑attention, optionally followed by a classical or quantum LSTM.
    The class can be instantiated as a drop‑in replacement for the original
    Conv filter, but exposes additional scaling knobs.
    """
    def __init__(self,
                 kernel_size: int = 2,
                 threshold: float = 0.0,
                 use_quantum: bool = False,
                 attention_type: str = 'classical',
                 autoenc: bool = True,
                 autoenc_cfg: AutoencoderConfig | None = None,
                 lstm_qubits: int = 0):
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.use_quantum = use_quantum
        self.attention_type = attention_type
        self.autoenc = autoenc
        self.lstm_qubits = lstm_qubits

        if use_quantum:
            raise NotImplementedError("Quantum path is provided in the QML module.")

        # Classical convolution
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

        # Optional auto‑encoder
        if autoenc:
            cfg = autoenc_cfg or AutoencoderConfig(
                input_dim=kernel_size * kernel_size,
                latent_dim=8,
                hidden_dims=(32, 16),
                dropout=0.1,
            )
            self.autoencoder = Autoencoder(
                input_dim=cfg.input_dim,
                latent_dim=cfg.latent_dim,
                hidden_dims=cfg.hidden_dims,
                dropout=cfg.dropout,
            )
        else:
            self.autoencoder = None

        # Self‑attention
        if attention_type == 'classical':
            self.attention = SelfAttention()
        else:
            self.attention = None

        # LSTM (classical or quantum)
        if lstm_qubits > 0:
            self.lstm = QLSTM(
                input_dim=kernel_size * kernel_size,
                hidden_dim=16,
                n_qubits=lstm_qubits,
            )
        else:
            self.lstm = nn.LSTM(
                input_size=kernel_size * kernel_size,
                hidden_size=16,
                batch_first=True,
            )

    def run(self, data: np.ndarray | torch.Tensor) -> float:
        """Apply the hybrid pipeline to a 2‑D input and return a scalar."""
        tensor = torch.as_tensor(data, dtype=torch.float32)
        tensor = tensor.view(1, 1, self.kernel_size, self.kernel_size)

        # Auto‑encoder
        if self.autoencoder:
            latent = self.autoencoder.encode(tensor)
        else:
            latent = tensor

        # Convolution
        conv_out = self.conv(latent)
        conv_vec = conv_out.view(-1)

        # Self‑attention
        if self.attention:
            rot = np.random.rand(3 * self.kernel_size * self.kernel_size)
            ent = np.random.rand(self.kernel_size * self.kernel_size - 1)
            attn_out = self.attention.run(rot, ent, conv_vec.numpy())
            conv_vec = torch.as_tensor(attn_out, dtype=torch.float32)
        else:
            conv_vec = conv_vec

        # LSTM
        seq = conv_vec.unsqueeze(0).unsqueeze(0)  # (batch, seq_len, features)
        if isinstance(self.lstm, nn.LSTM):
            lstm_out, _ = self.lstm(seq)
            output = lstm_out.mean().item()
        else:
            lstm_out, _ = self.lstm(seq)
            output = lstm_out.mean().item()

        return output

def Conv() -> ConvGen114:
    """Convenience factory matching the original API."""
    return ConvGen114()
    
__all__ = ["ConvGen114", "Conv"]
