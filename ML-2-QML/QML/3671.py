from __future__ import annotations

import strawberryfields as sf
from strawberryfields.ops import BSgate, Dgate, Kgate, Rgate, Sgate, MeasureFock
import torch
from dataclasses import dataclass
from typing import Sequence

# --------------------------------------------------------------------------- #
# Shared data structure for fraud‑detection parameters
# --------------------------------------------------------------------------- #
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

def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))

def _apply_layer(modes: Sequence, params: FraudLayerParameters, *, clip: bool) -> None:
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

# --------------------------------------------------------------------------- #
# Quantum hybrid circuit
# --------------------------------------------------------------------------- #
class FraudQuantumHybrid:
    """
    Photonic quantum circuit that processes 2×2 image patches with a
    fraud‑detection style variational layer and returns Fock counts.
    """
    def __init__(self, params: FraudLayerParameters) -> None:
        self.params = params

    def run(self, image: torch.Tensor, engine: sf.Engine) -> torch.Tensor:
        """
        Execute the circuit on a grayscale image and return a flattened
        vector of photon‑count samples.

        Parameters
        ----------
        image : torch.Tensor
            2‑D tensor of shape (H, W) with values in [0, 1].
        engine : sf.Engine
            Strawberry Fields engine (e.g. Engine('fock', backend_args={'cutoff_dim': 5})).

        Returns
        -------
        torch.Tensor
            1‑D tensor of sample counts.
        """
        H, W = image.shape
        assert H % 2 == 0 and W % 2 == 0, "Image dimensions must be even."
        patch_size = 2
        patches_per_dim = H // patch_size
        patch_count = patches_per_dim * patches_per_dim
        n_modes = 4 * patch_count

        prog = sf.Program(n_modes)
        with prog.context as q:
            for p in range(patches_per_dim):
                for q_ in range(patches_per_dim):
                    idx = p * patches_per_dim + q_
                    modes = [4 * idx + i for i in range(4)]
                    # encode pixel intensities as phase rotations
                    r = image[2 * p, 2 * q_]
                    s = image[2 * p, 2 * q_ + 1]
                    t = image[2 * p + 1, 2 * q_]
                    u = image[2 * p + 1, 2 * q_ + 1]
                    for i, val in enumerate([r, s, t, u]):
                        phase = float(val * torch.pi)
                        Rgate(phase) | q[modes[i]]
                    _apply_layer(modes, self.params, clip=True)

        # Measure all modes in Fock basis
        for i in range(n_modes):
            MeasureFock() | q[i]

        result = engine.run(prog)
        samples = result.samples
        return torch.tensor(samples, dtype=torch.float32).view(-1)

__all__ = ["FraudLayerParameters", "FraudQuantumHybrid"]
