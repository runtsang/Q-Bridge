import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Classical sub‑modules --------------------------------------------------------
class QuanvolutionFilter(nn.Module):
    """Classical convolutional filter mimicking a 2‑qubit quantum kernel."""
    def __init__(self, in_channels: int = 1, out_channels: int = 4,
                 kernel_size: int = 2, stride: int = 2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, 1, 28, 28)
        features = self.conv(x)                     # (batch, 4, 14, 14)
        return features.view(x.size(0), -1)         # (batch, 4*14*14)

class ClassicalSelfAttention(nn.Module):
    """Simplified self‑attention block over a sequence of embeddings."""
    def __init__(self, embed_dim: int = 4):
        super().__init__()
        self.embed_dim = embed_dim
        # Linear projections for query/key/value
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, embed_dim)
        Q = self.q_proj(x)          # (batch, seq_len, embed_dim)
        K = self.k_proj(x)          # (batch, seq_len, embed_dim)
        V = self.v_proj(x)          # (batch, seq_len, embed_dim)
        scores = torch.softmax(Q @ K.transpose(-2, -1) / np.sqrt(self.embed_dim),
                               dim=-1)                 # (batch, seq_len, seq_len)
        return scores @ V           # (batch, seq_len, embed_dim)

# Hybrid fraud detection model -----------------------------------------------
class FraudDetectionHybrid(nn.Module):
    """
    Classical + quantum hybrid fraud‑detection model.
    The forward pass uses a quanvolutional filter followed by self‑attention
    and a final linear head.  The quantum counterpart is exposed via
    `quantum_forward` which builds a Strawberry Fields program from the
    same parameters.
    """
    def __init__(self, num_classes: int = 2):
        super().__init__()
        # Feature extractor
        self.qfilter = QuanvolutionFilter()
        # Self‑attention over 14*14 = 196 patches, each of 4 dims
        self.attention = ClassicalSelfAttention(embed_dim=4)
        # Linear classification head
        self.classifier = nn.Linear(4 * 14 * 14, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, 1, 28, 28) grayscale images.
        Returns log‑softmax logits over fraud / non‑fraud.
        """
        # 1. Quanvolution
        features = self.qfilter(x)               # (batch, 4*14*14)
        # 2. Reshape for attention: (batch, 196, 4)
        seq_len = 14 * 14
        feats = features.view(x.size(0), seq_len, 4)
        # 3. Self‑attention
        attn_out = self.attention(feats)          # (batch, seq_len, 4)
        # 4. Flatten and classify
        flat = attn_out.view(x.size(0), -1)       # (batch, 4*14*14)
        logits = self.classifier(flat)           # (batch, num_classes)
        return F.log_softmax(logits, dim=-1)

    # -------------------------------------------------------------------------
    # Quantum interface
    # -------------------------------------------------------------------------
    @staticmethod
    def quantum_forward(inputs: torch.Tensor):
        """
        Run a Strawberry Fields photonic fraud‑detection circuit for each
        sample in `inputs`.  The circuit is parameterised by a simple
        fraud‑layer parameter set that mirrors the classical layer.
        Returns a list of measurement outcome counts.
        """
        from dataclasses import dataclass
        import strawberryfields as sf
        from strawberryfields.ops import BSgate, Dgate, Kgate, Rgate, Sgate

        @dataclass
        class FraudLayerParameters:
            bs_theta: float
            bs_phi: float
            phases: tuple[float, float]
            squeeze_r: tuple[float, float]
            squeeze_phi: tuple[float, float]
            displacement_r: tuple[float, float]
            displacement_phi: tuple[float, float]
            kerr: tuple[float, float]

        def _clip(val, bound):
            return max(-bound, min(bound, val))

        def _apply_layer(modes, params: FraudLayerParameters, clip: bool):
            BSgate(params.bs_theta, params.bs_phi) | (modes[0], modes[1])
            for i, phase in enumerate(params.phases):
                Rgate(phase) | modes[i]
            for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
                Sgate(r if not clip else _clip(r, 5), phi) | modes[i]
            BSgate(params.bs_theta, params.bs_phi) | (modes[0], modes[1])
            for i, phase in enumerate(params.phases):
                Rgate(phase) | modes[i]
            for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
                Dgate(r if not clip else _clip(r, 5), phi) | modes[i]
            for i, k in enumerate(params.kerr):
                Kgate(k if not clip else _clip(k, 1)) | modes[i]

        # Build a program per sample
        programs = []
        for _ in inputs:
            # Random but fixed parameters for demonstration
            params = FraudLayerParameters(
                bs_theta=0.5, bs_phi=0.3,
                phases=(0.1, -0.2),
                squeeze_r=(0.2, 0.3), squeeze_phi=(0.0, 0.0),
                displacement_r=(0.4, 0.5), displacement_phi=(0.1, -0.1),
                kerr=(0.05, 0.05)
            )
            prog = sf.Program(2)
            with prog.context as q:
                _apply_layer(q, params, clip=False)
            programs.append(prog)

        # Execute all programs on the local simulator
        results = []
        for prog in programs:
            eng = sf.Engine("gaussian")
            state = eng.run(prog).state
            # For simplicity we return the mean photon number per mode
            results.append(state.means)
        return results
