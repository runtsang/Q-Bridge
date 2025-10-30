"""Extended Polynomial ZZ Feature Map.

This module implements a feature map that generalises the
`zz_feature_map_poly` seed by adding optional three‑body
interactions, pre/post rotations, and a normalisation toggle.
The original API is preserved so the new map can be used
interchangeably with the seed variant.
"""

from __future__ import annotations

from math import pi
from typing import Callable, Sequence, Tuple, Mapping
import itertools

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterExpression, ParameterVector


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _resolve_entanglement(
    num_qubits: int,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]],
) -> list[Tuple[int, int]]:
    """Return a list of two‑qubit pairs according to a simple entanglement spec.

    Supported specs:
      - "full": all-to-all pairs (i < j)
      - "linear": nearest neighbors (0,1), (1,2),...
      - "circular": linear plus wrap‑around (n-1,0) if n > 2
      - explicit list of pairs like [(0, 2), (1, 3)]
      - callable: f(num_qubits) -> sequence of (i, j)
    """
    if isinstance(entanglement, str):
        if entanglement == "full":
            return [(i, j) for i in range(num_qubits) for j in range(i + 1, num_qubits)]
        if entanglement == "linear":
            return [(i, i + 1) for i in range(num_qubits - 1)]
        if entanglement == "circular":
            pairs = [(i, i + 1) for i in range(num_qubits - 1)]
            if num_qubits > 2:
                pairs.append((num_qubits - 1, 0))
            return pairs
        raise ValueError(f"Unknown entanglement string: {entanglement!r}")

    if callable(entanglement):
        pairs = list(entanglement(num_qubits))
        return [(int(i), int(j)) for (i, j) in pairs]

    # sequence of pairs
    pairs = [(int(i), int(j)) for (i, j) in entanglement]  # type: ignore[arg-type]
    # basic validation
    for (i, j) in pairs:
        if i == j:
            raise ValueError("Entanglement pairs must connect distinct qubits.")
        if not (0 <= i < num_qubits and 0 <= j < num_qubits):
            raise ValueError(f"Entanglement pair {(i, j)} out of range for n={num_qubits}.")
    return pairs


def _default_map_1(x: ParameterExpression) -> ParameterExpression:
    """Default φ1(x) = x."""
    return x


def _default_map_2(x: ParameterExpression, y: ParameterExpression) -> ParameterExpression:
    """Default φ2(x, y) = x · y."""
    return x * y


def _default_map_3(x: ParameterExpression, y: ParameterExpression, z: ParameterExpression) -> ParameterExpression:
    """Default φ3(x, y, z) = x · y · z."""
    return x * y * z


# ---------------------------------------------------------------------------
# Feature Map
# ---------------------------------------------------------------------------

def zz_feature_map_poly_extension(
    feature_dimension: int,
    *,
    reps: int = 2,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
    single_coeffs: Sequence[float] = (1.0,),
    pair_weight: float = 1.0,
    triplet_weight: float = 0.0,
    interaction_order: int = 2,
    basis: str = "h",
    parameter_prefix: str = "x",
    insert_barriers: bool = False,
    normalize: bool = False,
    pre_rotations: Mapping[int, Tuple[str, float]] | None = None,
    post_rotations: Mapping[int, Tuple[str, float]] | None = None,
    name: str | None = None,
) -> QuantumCircuit:
    """
    Extended polynomial ZZ feature map.

    This variant extends the original `ZZFeatureMapPoly` by adding optional
    three‑body interactions, pre/post rotations, and a normalisation toggle.
    It preserves the original API and can be used interchangeably with
    the seed variant.

    Parameters
    ----------
    feature_dimension : int
        Number of qubits / features.
    reps : int, default 2
        Number of repetitions.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]]
        Entanglement specification. Supported values are the same as in the seed.
    single_coeffs : Sequence[float], default (1.0,)
        Coefficients for the polynomial single‑qubit phase φ1(x) = Σ c_k · x^{k+1}.
    pair_weight : float, default 1.0
        Weight for the pairwise phase φ2(x, y) = pair_weight · x · y.
    triplet_weight : float, default 0.0
        Weight for the three‑body phase φ3(x, y, z) = triplet_weight · x · y · z.
    interaction_order : int, default 2
        2 for pairwise interactions only, 3 to include three‑body terms.
    basis : str, default "h"
        Basis preparation: "h" for Hadamard, "ry" for RY(π/2).
    parameter_prefix : str, default "x"
        Prefix for the ParameterVector.
    insert_barriers : bool, default False
        Insert barriers after each sub‑routine for easier circuit inspection.
    normalize : bool, default False
        If True, all feature values are scaled to the interval [0, π] before
        being used in the circuit.  The caller should provide features in the
        range [0, 1].
    pre_rotations : Mapping[int, Tuple[str, float]] | None, default None
        Optional pre‑rotations applied after the basis preparation and before
        the single‑qubit phases.  Keys are qubit indices; values are tuples
        `(rotation_type, angle)` where `rotation_type` is either `"rz"` or `"ry"`.
    post_rotations : Mapping[int, Tuple[str, float]] | None, default None
        Optional post‑rotations applied after the final repetition.
    name : str | None, default None
        Optional name for the resulting circuit.

    Returns
    -------
    QuantumCircuit
        A parameterised circuit ready for binding with a feature vector.

    Notes
    -----
    * The circuit exposes an ``input_params`` attribute containing the
      :class:`~qiskit.circuit.ParameterVector` for convenience.
    * The design choices are summarized in the module docstring and the
      function documentation.
    """
    if feature_dimension < 2:
        raise ValueError("feature_dimension must be >= 2.")
    if interaction_order not in (2, 3):
        raise ValueError("interaction_order must be 2 or 3.")
    if pair_weight < 0:
        raise ValueError("pair_weight must be non‑negative.")
    if triplet_weight < 0:
        raise ValueError("triplet_weight must be non‑negative.")
    if basis not in ("h", "ry"):
        raise ValueError("basis must be either 'h' or 'ry'.")

    n = int(feature_dimension)
    qc = QuantumCircuit(n, name=name or "ZZFeatureMapPolyExtension")

    # Parameter vector
    x = ParameterVector(parameter_prefix, n)

    # Scaling factor for optional normalisation
    scale = pi if normalize else 1.0

    # Helper maps
    def map1(xi: ParameterExpression) -> ParameterExpression:
        """φ1(x) = Σ c_k · x^{k+1}."""
        expr: ParameterExpression = 0 * xi
        p: ParameterExpression = xi
        for c in single_coeffs:
            expr = expr + c * p
            p = p * xi
        return scale * expr

    def map2(xi: ParameterExpression, xj: ParameterExpression) -> ParameterExpression:
        """φ2(x, y) = pair_weight · x · y."""
        return scale * pair_weight * xi * xj

    def map3(xi: ParameterExpression, xj: ParameterExpression, xk: ParameterExpression) -> ParameterExpression:
        """φ3(x, y, z) = triplet_weight · x · y · z."""
        return scale * triplet_weight * xi * xj * xk

    # Resolve entanglement pairs
    pairs = _resolve_entanglement(n, entanglement)

    # Pre‑rotation validation
    def _validate_rotations(rotations: Mapping[int, Tuple[str, float]] | None, role: str) -> None:
        if rotations is None:
            return
        for q, (rot, angle) in rotations.items():
            if not (0 <= q < n):
                raise ValueError(f"{role} rotation index {q} out of range for n={n}.")
            if rot not in ("rz", "ry"):
                raise ValueError(f"{role} rotation type must be 'rz' or 'ry', got {rot!r}.")

    _validate_rotations(pre_rotations, "pre")
    _validate_rotations(post_rotations, "post")

    for rep in range(int(reps)):
        # Basis preparation
        if basis == "h":
            qc.h(range(n))
        else:  # "ry"
            for q in range(n):
                qc.ry(pi / 2, q)

        # Pre‑rotations
        if pre_rotations:
            for q, (rot, angle) in pre_rotations.items():
                if rot == "rz":
                    qc.rz(angle, q)
                else:  # "ry"
                    qc.ry(angle, q)

        if insert_barriers:
            qc.barrier()

        # Single‑qubit phases
        for i in range(n):
            qc.p(2 * map1(x[i]), i)

        if insert_barriers:
            qc.barrier()

        # Pairwise ZZ interactions
        for (i, j) in pairs:
            angle = 2 * map2(x[i], x[j])
            qc.cx(i, j)
            qc.p(angle, j)
            qc.cx(i, j)

        # Three‑body interactions (if enabled)
        if interaction_order == 3:
            for (i, j, k) in itertools.combinations(range(n), 3):
                angle = 2 * map3(x[i], x[j], x[k])
                # To implement a controlled‑phase on three qubits:
                qc.cx(i, j)
                qc.cx(j, k)
                qc.p(angle, k)
                qc.cx(j, k)
                qc.cx(i, j)

        if insert_barriers and rep!= reps - 1:
            qc.barrier()

    # Post‑rotations
    if post_rotations:
        for q, (rot, angle) in post_rotations.items():
            if rot == "rz":
                qc.rz(angle, q)
            else:  # "ry"
                qc.ry(angle, q)

    qc.input_params = x  # type: ignore[attr-defined]
    return qc


class ZZFeatureMapPolyExtension(QuantumCircuit):
    """
    Class‑style wrapper for the extended polynomial ZZ feature map.

    Parameters are identical to :func:`zz_feature_map_poly_extension`.
    """

    def __init__(
        self,
        feature_dimension: int,
        *,
        reps: int = 2,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
        single_coeffs: Sequence[float] = (1.0,),
        pair_weight: float = 1.0,
        triplet_weight: float = 0.0,
        interaction_order: int = 2,
        basis: str = "h",
        parameter_prefix: str = "x",
        insert_barriers: bool = False,
        normalize: bool = False,
        pre_rotations: Mapping[int, Tuple[str, float]] | None = None,
        post_rotations: Mapping[int, Tuple[str, float]] | None = None,
        name: str = "ZZFeatureMapPolyExtension",
    ) -> None:
        built = zz_feature_map_poly_extension(
            feature_dimension,
            reps=reps,
            entanglement=entanglement,
            single_coeffs=single_coeffs,
            pair_weight=pair_weight,
            triplet_weight=triplet_weight,
            interaction_order=interaction_order,
            basis=basis,
            parameter_prefix=parameter_prefix,
            insert_barriers=insert_barriers,
            normalize=normalize,
            pre_rotations=pre_rotations,
            post_rotations=post_rotations,
            name=name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)
        self.input_params = built.input_params  # type: ignore[attr-defined]


__all__ = ["ZZFeatureMapPolyExtension", "zz_feature_map_poly_extension"]
