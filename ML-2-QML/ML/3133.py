import torch
import torch.nn as nn

class HybridFunction(torch.autograd.Function):
    """Differentiable sigmoid activation that mimics a quantum expectation head."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, shift: float) -> torch.Tensor:
        outputs = torch.sigmoid(inputs + shift)
        ctx.save_for_backward(outputs)
        return outputs

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (outputs,) = ctx.saved_tensors
        grad_inputs = grad_output * outputs * (1 - outputs)
        return grad_inputs, None

class AutoencoderConfig:
    """Configuration values for :class:`AutoencoderNet`."""
    def __init__(self, input_dim: int, latent_dim: int = 32, hidden_dims: tuple[int, int] = (128, 64), dropout: float = 0.1):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout

class AutoencoderNet(nn.Module):
    """A lightweight multilayer perceptron autoencoder."""
    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        encoder_layers = []
        in_dim = config.input_dim
        for hidden in config.hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, hidden))
            encoder_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                encoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        encoder_layers.append(nn.Linear(in_dim, config.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        in_dim = config.latent_dim
        for hidden in reversed(config.hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, hidden))
            decoder_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                decoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        decoder_layers.append(nn.Linear(in_dim, config.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.encoder(inputs)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(latents)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(inputs))

def Autoencoder(input_dim: int, *, latent_dim: int = 32, hidden_dims: tuple[int, int] = (128, 64), dropout: float = 0.1) -> AutoencoderNet:
    """Factory that returns a configured autoencoder."""
    config = AutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
    )
    return AutoencoderNet(config)

class AutoEncoderHybridNet(nn.Module):
    """Hybrid classifier that first compresses the input with an autoencoder
    and then classifies using a small MLP followed by a differentiable
    sigmoid head (classical or quantum)."""
    def __init__(self, input_dim: int, latent_dim: int = 64, hidden_dims=(32, 16), shift: float = 0.0) -> None:
        super().__init__()
        self.autoencoder = Autoencoder(input_dim, latent_dim=latent_dim)
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dims[1], 1),
        )
        self.hybrid = HybridFunction.apply

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs.view(inputs.size(0), -1)
        latent = self.autoencoder.encode(x)
        logits = self.classifier(latent)
        probs = self.hybrid(logits, 0.0)
        return torch.cat((probs, 1 - probs), dim=-1)

__all__ = ["AutoEncoderHybridNet", "Autoencoder", "AutoencoderNet", "AutoencoderConfig", "HybridFunction"]
