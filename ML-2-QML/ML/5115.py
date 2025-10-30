import math
import torch
from torch import nn
import torch.nn.functional as F

class AutoencoderConfig:
    """Configuration for the quantum-inspired autoencoder."""
    def __init__(self, input_dim, latent_dim=32, hidden_dims=(128,64), dropout=0.1):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout

class QuantumInspiredAutoencoder(nn.Module):
    """Autoencoder with sine activations to emulate quantum behaviour."""
    def __init__(self, config: AutoencoderConfig):
        super().__init__()
        encoder_layers = []
        in_dim = config.input_dim
        for hidden in config.hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, hidden))
            encoder_layers.append(nn.Sigmoid())  # quantum-inspired activation
            if config.dropout > 0.0:
                encoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        encoder_layers.append(nn.Linear(in_dim, config.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        in_dim = config.latent_dim
        for hidden in reversed(config.hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, hidden))
            decoder_layers.append(nn.Sigmoid())
            if config.dropout > 0.0:
                decoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        decoder_layers.append(nn.Linear(in_dim, config.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

class QuantumInspiredConv(nn.Module):
    """2‑D filter that mimics a quantum quanvolution."""
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0):
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)
        # initialise with sinusoidal weights
        with torch.no_grad():
            theta = torch.rand(self.conv.weight.shape) * 2 * math.pi
            self.conv.weight.copy_(torch.sin(theta))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Expected input shape: (batch, 1, H, W).
        Returns a tensor of the same spatial shape with a sigmoid activation.
        """
        return torch.sigmoid(self.conv(x) - self.threshold)

class QuantumInspiredAttention(nn.Module):
    """Classical attention block with quantum‑style activations."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.out_linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()
        q = self.q_linear(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        k = self.k_linear(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_linear(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        return self.out_linear(out)

class QuantumInspiredFeedForward(nn.Module):
    """Feed‑forward block with sine activation."""
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.sin(self.linear1(x))))

class QuantumInspiredTransformerBlock(nn.Module):
    """Transformer block that uses the quantum‑inspired attention and feed‑forward."""
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = QuantumInspiredAttention(embed_dim, num_heads, dropout)
        self.ffn = QuantumInspiredFeedForward(embed_dim, ffn_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

class HybridQuantumLayer(nn.Module):
    """
    Composite module that stitches together quantum‑inspired convolution, auto‑encoder,
    transformer blocks and a fully‑connected layer.
    """
    def __init__(
        self,
        n_features: int = 1,
        embed_dim: int = 64,
        num_heads: int = 4,
        ffn_dim: int = 128,
        num_blocks: int = 2,
        use_quantum_fcl: bool = True,
        use_quantum_transformer: bool = True,
        use_quantum_conv: bool = True,
        use_quantum_autoencoder: bool = True,
        device: str | torch.device = "cpu",
    ):
        super().__init__()
        self.device = torch.device(device)
        self.use_quantum_fcl = use_quantum_fcl
        self.use_quantum_transformer = use_quantum_transformer
        self.use_quantum_conv = use_quantum_conv
        self.use_quantum_autoencoder = use_quantum_autoencoder

        if self.use_quantum_conv:
            self.conv = QuantumInspiredConv(kernel_size=2, threshold=0.0).to(self.device)
        else:
            self.conv = None

        if self.use_quantum_autoencoder:
            cfg = AutoencoderConfig(input_dim=n_features, latent_dim=32, hidden_dims=(128, 64), dropout=0.1)
            self.autoencoder = QuantumInspiredAutoencoder(cfg).to(self.device)
        else:
            self.autoencoder = None

        if self.use_quantum_transformer:
            self.transformer = nn.Sequential(
                *[QuantumInspiredTransformerBlock(embed_dim, num_heads, ffn_dim) for _ in range(num_blocks)]
            ).to(self.device)
        else:
            self.transformer = None

        if self.use_quantum_fcl:
            self.fcl = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.Sigmoid()
            ).to(self.device)
        else:
            self.fcl = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the hybrid network.
        - If conv is enabled, the input should be 4‑D (batch, 1, H, W).
        - If autoencoder is enabled, the input is flattened before encoding.
        """
        if self.conv is not None:
            x = self.conv(x)  # shape (batch, 1, H, W)
            x = x.view(x.size(0), -1)  # flatten

        if self.autoencoder is not None:
            x = self.autoencoder.encode(x)

        if self.transformer is not None:
            x = self.transformer(x)

        if self.fcl is not None:
            x = self.fcl(x)

        return x

    def run(self, data):
        """
        Convenience wrapper that accepts a NumPy array, forwards it through the network
        and returns a NumPy array of outputs.
        """
        with torch.no_grad():
            tensor = torch.as_tensor(data, dtype=torch.float32, device=self.device)
            output = self.forward(tensor)
        return output.cpu().numpy()

__all__ = [
    "AutoencoderConfig",
    "QuantumInspiredAutoencoder",
    "QuantumInspiredConv",
    "QuantumInspiredAttention",
    "QuantumInspiredFeedForward",
    "QuantumInspiredTransformerBlock",
    "HybridQuantumLayer",
]
