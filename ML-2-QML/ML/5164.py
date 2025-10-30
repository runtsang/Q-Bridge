from __future__ import annotations

import torch
from torch import nn
from typing import Iterable, Tuple
from dataclasses import dataclass

@dataclass
class FraudLayerParameters:
    bs_theta: float
    bs_phi: float
    phases: Tuple[float, float]
    squeeze_r: Tuple[float, float]
    squeeze_phi: Tuple[float, float]
    displacement_r: Tuple[float, float]
    displacement_phi: Tuple[float, float]
    kerr: Tuple[float, float]
    dropout: float = 0.0

def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))

def _layer_from_params(params: FraudLayerParameters, *, clip: bool) -> nn.Module:
    weight = torch.tensor(
        [
            [params.bs_theta, params.bs_phi],
            [params.squeeze_r[0], params.squeeze_r[1]],
        ],
        dtype=torch.float32,
    )
    bias = torch.tensor(params.phases, dtype=torch.float32)
    if clip:
        weight = weight.clamp(-5.0, 5.0)
        bias = bias.clamp(-5.0, 5.0)
    linear = nn.Linear(2, 2)
    with torch.no_grad():
        linear.weight.copy_(weight)
        linear.bias.copy_(bias)
    activation = nn.Tanh()
    scale = torch.tensor(params.displacement_r, dtype=torch.float32)
    shift = torch.tensor(params.displacement_phi, dtype=torch.float32)

    class Layer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = linear
            self.activation = activation
            self.register_buffer("scale", scale)
            self.register_buffer("shift", shift)
            self.drop = nn.Dropout(params.dropout) if params.dropout > 0.0 else nn.Identity()

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            outputs = self.activation(self.linear(inputs))
            outputs = outputs * self.scale + self.shift
            return self.drop(outputs)

    return Layer()

class AutoencoderConfig:
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1

class AutoencoderNet(nn.Module):
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

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.decode(self.encode(inputs))

def Autoencoder(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int, int] = (128, 64),
    dropout: float = 0.1,
) -> AutoencoderNet:
    config = AutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
    )
    return AutoencoderNet(config)

class FraudNetHybrid(nn.Module):
    def __init__(
        self,
        autoencoder_config: AutoencoderConfig,
        fraud_params: FraudLayerParameters,
        n_fraud_layers: int = 2,
        use_qlstm: bool = False,
        qlstm_hidden_dim: int = 64,
        qlstm_n_qubits: int = 0,
    ) -> None:
        super().__init__()
        self.autoencoder = Autoencoder(autoencoder_config)
        self.fraud_layers = nn.ModuleList(
            [_layer_from_params(fraud_params, clip=False)] +
            [_layer_from_params(fraud_params, clip=True) for _ in range(n_fraud_layers)]
        )
        self.final_linear = nn.Linear(2, 1)
        if use_qlstm:
            class QLSTM(nn.Module):
                def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int) -> None:
                    super().__init__()
                    self.input_dim = input_dim
                    self.hidden_dim = hidden_dim
                    self.n_qubits = n_qubits
                    gate_dim = hidden_dim
                    self.forget_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
                    self.input_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
                    self.update_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
                    self.output_linear = nn.Linear(input_dim + hidden_dim, gate_dim)

                def forward(
                    self,
                    inputs: torch.Tensor,
                    states: Tuple[torch.Tensor, torch.Tensor] | None = None,
                ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:  # type: ignore[override]
                    hx, cx = self._init_states(inputs, states)
                    outputs = []
                    for x in inputs.unbind(dim=0):
                        combined = torch.cat([x, hx], dim=1)
                        f = torch.sigmoid(self.forget_linear(combined))
                        i = torch.sigmoid(self.input_linear(combined))
                        g = torch.tanh(self.update_linear(combined))
                        o = torch.sigmoid(self.output_linear(combined))
                        cx = f * cx + i * g
                        hx = o * torch.tanh(cx)
                        outputs.append(hx.unsqueeze(0))
                    stacked = torch.cat(outputs, dim=0)
                    return stacked, (hx, cx)

                def _init_states(
                    self,
                    inputs: torch.Tensor,
                    states: Tuple[torch.Tensor, torch.Tensor] | None,
                ) -> Tuple[torch.Tensor, torch.Tensor]:
                    if states is not None:
                        return states
                    batch_size = inputs.size(1)
                    device = inputs.device
                    hx = torch.zeros(batch_size, self.hidden_dim, device=device)
                    cx = torch.zeros(batch_size, self.hidden_dim, device=device)
                    return hx, cx

            self.qlstm = QLSTM(input_dim=2, hidden_dim=qlstm_hidden_dim, n_qubits=qlstm_n_qubits)
        else:
            self.qlstm = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        latent = self.autoencoder.encode(x)
        out = latent
        for layer in self.fraud_layers:
            out = layer(out)
        out = self.final_linear(out)
        if self.qlstm is not None:
            out, _ = self.qlstm(out.unsqueeze(0))
            out = out.squeeze(0)
        return out
