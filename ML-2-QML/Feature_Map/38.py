"""Extended polynomial ZZ feature map with higher‑order interactions and optional pre/post rotations.

The module provides a function and a QuantumCircuit subclass that both build a
feature‑map circuit capable of encoding classical data into quantum phase
information.  It is a direct extension of the original `ZZFeatureMapPoly`
module: the same basis‑prep and two‑qubit phase structure are preserved while
additional interaction terms and normalisation options are introduced.

Key extensions
--------------
- **Higher‑order ZZ interactions** up to a user‑specified maximum order.
- **Optional data normalisation**: a global scaling factor applied to all
  phase angles.
- **Pre‑ and post‑basis rotations** (Hadamard or RY(π/2)) that can be toggled.
- **Per‑repetition entanglement specification** allowing different
  coupling patterns across layers.
- **Barrier insertion** for visual debugging.

The interface is intentionally backwards compatible: the function
`zz_feature_map_poly_extended` accepts the same core arguments as the seed
module and adds a handful of optional flags.  Users can also instantiate the
`ZZFeatureMapPolyExtended` class directly.

Typical usage
-------------
```python
from zz_feature_map_poly_extended import zz_feature_map_poly_extended

qc = zz_feature_map_poly_extended(
    feature_dimension=4,
    reps=3,
    entanglement="full",
    single_coeffs=(1.0, 0.5),
    pair_weight=1.0,
    higher_order_weights=(0.8, ),
    use_pre_rot=True,
    use_post_rot=False,
    use_normalization=True,
    normalization_factor=pi/2,
)
```
"""

from __future__ import annotations

import itertools
from math import pi
from typing import Callable, Iterable, List, Optional, Sequence, Tuple, Union

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterExpression, ParameterVector


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _resolve_entanglement(
    num_qubits: int,
    entanglement: Union[
        str,
        Sequence[Tuple[int, int]],
        Callable[[int], Sequence[Tuple[int, int]]],
        List[Union[str, Sequence[Tuple[int, int]], Callable[[int], Sequence[Tuple[int, int]]]]],
    ],
    reps: int,
) -> List[List[Tuple[int, int]]]:
    """
    Resolve the entanglement specification into a list of pair lists, one per
    repetition.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] | list
        Specification of the two‑qubit coupling pattern.  It can be:
        * a string ("full", "linear", "circular")
        * an explicit pair list
        * a callable that returns a pair list given the number of qubits
        * a list of the above, one element per repetition
    reps : int
        Number of repetitions.

    Returns
    -------
    List[List[Tuple[int, int]]]
        A list of pair lists, one per repetition.

    Raises
    ------
    ValueError
        If the specification is invalid or the list length does not match reps.
    """
    def _resolve_one(spec: Union[str, Sequence[Tuple[int, int]], Callable[[int], Sequence[Tuple[int, int]]]]) -> List[Tuple[int, int]]:
        if isinstance(spec, str):
            if spec == "full":
                return [(i, j) for i in range(num_qubits) for j in range(i + 1, num_qubits)]
            if spec == "linear":
                return [(i, i + 1) for i in range(num_qubits - 1)]
            if spec == "circular":
                pairs = [(i, i + 1) for i in range(num_qubits - 1)]
                if num_qubits > 2:
                    pairs.append((num_qubits - 1, 0))
                return pairs
            raise ValueError(f"Unknown entanglement string: {spec!r}")
        if callable(spec):
            pairs = list(spec(num_qubits))
            return [(int(i), int(j)) for (i, j) in pairs]
        # explicit sequence
        pairs = [(int(i), int(j)) for (i, j) in spec]  # type: ignore[arg-type]
        for (i, j) in pairs:
            if i == j:
                raise ValueError("Entanglement pairs must connect distinct qubits.")
            if not (0 <= i < num_qubits and 0 <= j < num_qubits):
                raise ValueError(f"Entanglement pair {(i, j)} out of range for n={num_qubits}.")
        return pairs

    # Handle per‑repetition list
    if isinstance(entanglement, list):
        if len(entanglement)!= reps:
            raise ValueError("Length of entanglement list must match number of repetitions.")
        return [_resolve_one(spec) for spec in entanglement]
    else:
        # Same spec for all repetitions
        return [_resolve_one(entanglement) for _ in range(reps)]


# ---------------------------------------------------------------------------
# Default mapping functions
# ---------------------------------------------------------------------------

def _default_map1(x: ParameterExpression, coeffs: Sequence[float]) -> ParameterExpression:
    """Default single‑qubit polynomial φ1(x) = Σ_k coeffs[k] · x^(k+1)."""
    expr: ParameterExpression = 0
    power: ParameterExpression = x
    for c in coeffs:
        expr = expr + c * power
        power = power * x
    return expr


def _default_map2(x: ParameterExpression, y: ParameterExpression, weight: float) -> ParameterExpression:
    """Default pairwise interaction φ2(x, y) = weight · x · y."""
    return weight * x * y


def _default_map_k(xs: Sequence[ParameterExpression], weight: float) -> ParameterExpression:
    """Default higher‑order interaction φk(x1,..,xk) = weight · Π_i xi."""
    prod: ParameterExpression = 1
    for xi in xs:
        prod = prod * xi
    return weight * prod


# ---------------------------------------------------------------------------
# Feature‑map builder
# ---------------------------------------------------------------------------

def zz_feature_map_poly_extended(
    feature_dimension: int,
    reps: int = 2,
    entanglement: Union[
        str,
        Sequence[Tuple[int, int]],
        Callable[[int], Sequence[Tuple[int, int]]],
        List[Union[str, Sequence[Tuple[int, int]], Callable[[int], Sequence[Tuple[int, int]]]]],
    ] = "full",
    single_coeffs: Sequence[float] = (1.0,),
    pair_weight: float = 1.0,
    higher_order_weights: Sequence[float] = (1.0,),
    max_interaction_order: int = 2,
    basis: str = "h",  # "h" or "ry"
    parameter_prefix: str = "x",
    insert_barriers: bool = False,
    use_pre_rot: bool = True,
    use_post_rot: bool = True,
    use_normalization: bool = False,
    normalization_factor: float = 1.0,
    name: Optional[str] = None,
) -> QuantumCircuit:
    """
    Build an extended polynomial ZZ feature map.

    Parameters
    ----------
    feature_dimension : int
        Number of classical features (must be >= 2).
    reps : int
        Number of repetitions (layers) of the encoding.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] | list
        Two‑qubit coupling pattern.  If a list is supplied, it must contain
        one specification per repetition.
    single_coeffs : Sequence[float]
        Coefficients for the single‑qubit polynomial φ1(x).
    pair_weight : float
        Weight applied to the pairwise interaction φ2(x, y).
    higher_order_weights : Sequence[float]
        Weights for interactions of order k>2.  The i‑th element corresponds to
        order (i+3).  Its length must be at least ``max_interaction_order-2``.
    max_interaction_order : int
        Maximum order of interactions to include (>= 2).  Order 2 corresponds
        to the original ZZ pairwise term.
    basis : str
        Basis preparation before phase encoding: "h" for Hadamard, "ry" for
        RY(π/2).
    parameter_prefix : str
        Prefix for the ParameterVector naming.
    insert_barriers : bool
        Insert barriers between logical blocks for visual clarity.
    use_pre_rot : bool
        If True, apply the chosen basis preparation before the single‑qubit
        phases and higher‑order interactions.
    use_post_rot : bool
        If True, apply the chosen basis preparation after all interactions.
    use_normalization : bool
        If True, multiply all phase angles by ``normalization_factor``.
    normalization_factor : float
        Global scaling factor applied to all phase angles when
        ``use_normalization`` is True.
    name : str | None
        Optional circuit name.

    Returns
    -------
    QuantumCircuit
        The constructed feature‑map circuit.  The attribute ``input_params``
        contains the ParameterVector that should be bound to classical data.

    Raises
    ------
    ValueError
        On invalid argument combinations.
    """
    # Argument validation
    if feature_dimension < 2:
        raise ValueError("feature_dimension must be >= 2.")
    if reps < 1:
        raise ValueError("reps must be >= 1.")
    if max_interaction_order < 2:
        raise ValueError("max_interaction_order must be >= 2.")
    if len(higher_order_weights) < max_interaction_order - 2:
        raise ValueError(
            f"higher_order_weights must contain at least {max_interaction_order - 2} "
            "elements (one per order > 2)."
        )
    if basis not in ("h", "ry"):
        raise ValueError("basis must be 'h' or 'ry'.")
    if not isinstance(use_pre_rot, bool) or not isinstance(use_post_rot, bool):
        raise ValueError("use_pre_rot and use_post_rot must be boolean.")
    if use_normalization:
        if normalization_factor <= 0:
            raise ValueError("normalization_factor must be positive when use_normalization is True.")

    # Resolve entanglement patterns per repetition
    pair_lists = _resolve_entanglement(feature_dimension, entanglement, reps)

    qc = QuantumCircuit(feature_dimension, name=name or "ZZFeatureMapPolyExtended")
    x = ParameterVector(parameter_prefix, feature_dimension)

    # Global scaling factor
    scale = normalization_factor if use_normalization else 1.0

    # Pre‑basis rotation
    def _apply_basis(qc_inner: QuantumCircuit) -> None:
        if basis == "h":
            qc_inner.h(range(feature_dimension))
        else:  # ry
            for q in range(feature_dimension):
                qc_inner.ry(pi / 2, q)

    for rep in range(reps):
        if use_pre_rot:
            _apply_basis(qc)

        if insert_barriers:
            qc.barrier()

        # Single‑qubit phases
        for i in range(feature_dimension):
            angle = 2 * scale * _default_map1(x[i], single_coeffs)
            qc.p(angle, i)

        if insert_barriers:
            qc.barrier()

        # Pairwise ZZ interactions (order 2)
        for (i, j) in pair_lists[rep]:
            angle = 2 * scale * _default_map2(x[i], x[j], pair_weight)
            qc.cx(i, j)
            qc.p(angle, j)
            qc.cx(i, j)

        # Higher‑order interactions (order >=3)
        for order in range(3, max_interaction_order + 1):
            weight = higher_order_weights[order - 3]
            for combo in itertools.combinations(range(feature_dimension), order):
                angle = 2 * scale * _default_map_k([x[q] for q in combo], weight)
                # Multi‑controlled Z via chain of CNOTs
                for ctrl in combo[:-1]:
                    qc.cx(ctrl, combo[-1])
                qc.p(angle, combo[-1])
                for ctrl in reversed(combo[:-1]):
                    qc.cx(ctrl, combo[-1])

        if insert_barriers and rep!= reps - 1:
            qc.barrier()

    if use_post_rot:
        _apply_basis(qc)

    qc.input_params = x  # type: ignore[attr-defined]
    return qc


# ---------------------------------------------------------------------------
# OO wrapper
# ---------------------------------------------------------------------------

class ZZFeatureMapPolyExtended(QuantumCircuit):
    """QuantumCircuit subclass wrapping :func:`zz_feature_map_poly_extended`."""

    def __init__(
        self,
        feature_dimension: int,
        reps: int = 2,
        entanglement: Union[
            str,
            Sequence[Tuple[int, int]],
            Callable[[int], Sequence[Tuple[int, int]]],
            List[Union[str, Sequence[Tuple[int, int]], Callable[[int], Sequence[Tuple[int, int]]]]],
        ] = "full",
        single_coeffs: Sequence[float] = (1.0,),
        pair_weight: float = 1.0,
        higher_order_weights: Sequence[float] = (1.0,),
        max_interaction_order: int = 2,
        basis: str = "h",
        parameter_prefix: str = "x",
        insert_barriers: bool = False,
        use_pre_rot: bool = True,
        use_post_rot: bool = True,
        use_normalization: bool = False,
        normalization_factor: float = 1.0,
        name: str = "ZZFeatureMapPolyExtended",
    ) -> None:
        built = zz_feature_map_poly_extended(
            feature_dimension,
            reps,
            entanglement,
            single_coeffs,
            pair_weight,
            higher_order_weights,
            max_interaction_order,
            basis,
            parameter_prefix,
            insert_barriers,
            use_pre_rot,
            use_post_rot,
            use_normalization,
            normalization_factor,
            name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)
        self.input_params = built.input_params  # type: ignore[attr-defined]


__all__ = ["ZZFeatureMapPolyExtended", "zz_feature_map_poly_extended"]
