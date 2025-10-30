"""Hybrid photonic QCNN for fraud detection.

This module implements a Strawberry Fields program that combines
the photonic fraud detection circuit with a QCNN‑style convolution
and pooling structure.  Each convolutional block applies a two‑mode
unitary consisting of a balanced beamsplitter, phase shifters,
squeezers, displacements, and Kerr gates, mirroring the
parameterised photonic layer.  Pooling reduces the number of
modes by discarding or combining them, analogous to the
classical pooling layers in the hybrid model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import strawberryfields as sf
from strawberryfields.ops import BSgate, Dgate, Kgate, Rgate, Sgate


@dataclass
class FraudLayerParameters:
    """Parameters for a photonic‑inspired two‑mode unitary."""
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
    """Apply a two‑mode photonic layer to the given modes."""
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


def _conv_block(mode_pairs: Sequence[tuple[int, int]], params_list: Iterable[FraudLayerParameters]) -> None:
    """Apply a convolutional block to each pair of modes."""
    for (m1, m2), params in zip(mode_pairs, params_list):
        _apply_layer((m1, m2), params, clip=True)


def _pool_block(source_modes: Sequence[int], sink_modes: Sequence[int], params_list: Iterable[FraudLayerParameters]) -> None:
    """Apply a pooling block that combines source modes into sink modes."""
    for src, sink, params in zip(source_modes, sink_modes, params_list):
        _apply_layer((src, sink), params, clip=True)


def build_fraud_detection_quantum(num_modes: int = 8) -> sf.Program:
    """Construct a QCNN‑style photonic program for fraud detection."""
    program = sf.Program(num_modes)
    with program.context as q:
        # Initial photonic layer on all modes
        init_params = FraudLayerParameters(
            bs_theta=0.0,
            bs_phi=0.0,
            phases=(0.0, 0.0),
            squeeze_r=(0.0, 0.0),
            squeeze_phi=(0.0, 0.0),
            displacement_r=(0.0, 0.0),
            displacement_phi=(0.0, 0.0),
            kerr=(0.0, 0.0),
        )
        _apply_layer((q[0], q[1]), init_params, clip=False)

        # Stage 1: convolution on four pairs
        conv1_params = [
            FraudLayerParameters(
                bs_theta=0.1, bs_phi=0.2,
                phases=(0.3, 0.4),
                squeeze_r=(0.5, 0.6),
                squeeze_phi=(0.7, 0.8),
                displacement_r=(0.9, 1.0),
                displacement_phi=(1.1, 1.2),
                kerr=(1.3, 1.4),
            )
            for _ in range(4)
        ]
        _conv_block([(0, 1), (2, 3), (4, 5), (6, 7)], conv1_params)

        # Stage 1 pooling: combine into four modes
        pool1_params = [
            FraudLayerParameters(
                bs_theta=0.2, bs_phi=0.3,
                phases=(0.4, 0.5),
                squeeze_r=(0.6, 0.7),
                squeeze_phi=(0.8, 0.9),
                displacement_r=(1.0, 1.1),
                displacement_phi=(1.2, 1.3),
                kerr=(1.4, 1.5),
            )
            for _ in range(4)
        ]
        _pool_block([0, 2, 4, 6], [1, 3, 5, 7], pool1_params)

        # Stage 2: convolution on two pairs
        conv2_params = [
            FraudLayerParameters(
                bs_theta=0.3, bs_phi=0.4,
                phases=(0.5, 0.6),
                squeeze_r=(0.7, 0.8),
                squeeze_phi=(0.9, 1.0),
                displacement_r=(1.1, 1.2),
                displacement_phi=(1.3, 1.4),
                kerr=(1.5, 1.6),
            )
            for _ in range(2)
        ]
        _conv_block([(1, 3), (5, 7)], conv2_params)

        # Stage 2 pooling: reduce to two modes
        pool2_params = [
            FraudLayerParameters(
                bs_theta=0.4, bs_phi=0.5,
                phases=(0.6, 0.7),
                squeeze_r=(0.8, 0.9),
                squeeze_phi=(1.0, 1.1),
                displacement_r=(1.2, 1.3),
                displacement_phi=(1.4, 1.5),
                kerr=(1.6, 1.7),
            )
            for _ in range(2)
        ]
        _pool_block([1, 5], [3, 7], pool2_params)

        # Stage 3: convolution on the remaining pair
        conv3_params = [
            FraudLayerParameters(
                bs_theta=0.5, bs_phi=0.6,
                phases=(0.7, 0.8),
                squeeze_r=(0.9, 1.0),
                squeeze_phi=(1.1, 1.2),
                displacement_r=(1.3, 1.4),
                displacement_phi=(1.5, 1.6),
                kerr=(1.7, 1.8),
            )
        ]
        _conv_block([(3, 7)], conv3_params)

        # Final pooling: collapse to a single mode
        pool3_params = [
            FraudLayerParameters(
                bs_theta=0.6, bs_phi=0.7,
                phases=(0.8, 0.9),
                squeeze_r=(1.0, 1.1),
                squeeze_phi=(1.2, 1.3),
                displacement_r=(1.4, 1.5),
                displacement_phi=(1.6, 1.7),
                kerr=(1.8, 1.9),
            )
        ]
        _pool_block([3], [7], pool3_params)

    return program
