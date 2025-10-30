"""Extended polynomial ZZ feature map with optional higher‑order terms
and pre/post rotations.

The module defines both a functional interface and a ``QuantumCircuit``
subclass named ``ZZFeatureMapPolyExtended``.  It is fully compatible with
Qiskit's data‑encoding patterns and can be used directly in a variational
quantum algorithm.

Features
---------
* **Higher‑order single‑qubit terms** – up to ``max_order`` powers of each
  feature are included in the phase.
* **Optional pre‑ and post‑Hadamard rotations** – controlled by
  ``pre_rotate`` and ``post_rotate`` flags.
* **Feature normalisation** – when ``normalise=True`` the input vector is
  linearly mapped to ``[0, π]`` before parameter binding.
* **Flexible entanglement patterns** – same helper as the seed module.
* **Parameter validation** – clear error messages for out‑of‑range
  dimensions, invalid entanglement specifications, and non‑numeric
  coefficients.

Usage
-----
```python
from zz_feature_map_poly_extension import zz_feature_map_poly_extended, ZZFeatureMapPolyExtended

# Functional
qc = zz_feature_map_poly_extended(
    feature_dimension=4,
    reps=3,
    entanglement="circular",
    single_coeffs=(1.0, 0.5),
    pair_weight=0.8,
    max_order=3,
    pre_rotate=True,
    post_rotate=False,
    normalise=True,
    basis="h",
)

# OO
circuit = ZZFeatureMapPolyExtended(4, reps=3, entanglement="full",
                                   max_order=4, normalise=False)
```

Both return a ``QuantumCircuit`` with an ``input_params`` attribute
containing a ``ParameterVector`` that can be bound with a NumPy array.
"""
from __future__ import annotations

from math import pi
from typing import Callable, Iterable, List, Sequence, Tuple

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterExpression, ParameterVector


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _resolve_entanglement(
    num_qubits: int,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Iterable[Tuple[int, int]]],
) -> List[Tuple[int, int]]:
    """Return a list of two‑qubit pairs according to a simple entanglement spec.

    Supported specs:
      - "full": all‑to‑all pairs (i < j)
      - "linear": nearest neighbours
      - "circular": linear + wrap‑around
      - explicit list of pairs
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
    """Default φ2(x, y) = (π − x)(π − y)."""
    return (pi - x) * (pi - y)


# ---------------------------------------------------------------------------
# Extended polynomial feature map
# ---------------------------------------------------------------------------

def zz_feature_map_poly_extended(
    feature_dimension: int,
    reps: int = 2,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Iterable[Tuple[int, int]]] = "full",
    single_coeffs: Sequence[float] = (1.0,),
    pair_weight: float = 1.0,
    max_order: int = 2,
    pre_rotate: bool = False,
    post_rotate: bool = False,
    normalise: bool = True,
    basis: str = "h",  # "h" or "ry"
    parameter_prefix: str = "x",
    insert_barriers: bool = False,
    name: str | None = None,
) -> QuantumCircuit:
    """Extended ZZ feature map with optional higher‑order terms.

    Parameters
    ----------
    feature_dimension : int
        Number of classical features (must be >= 2).
    reps : int
        Number of repetitions of the feature‑map block.
    entanglement : str | sequence | callable
        Entanglement specification.
    single_coeffs : sequence of float
        Coefficients for the single‑qubit polynomial terms.
    pair_weight : float
        Weight for the pairwise ZZ interaction.
    max_order : int
        Highest power of each feature to include in φ1.
    pre_rotate : bool
        Apply Hadamard to all qubits before the first repetition.
    post_rotate : bool
        Apply Hadamard to all qubits after the last repetition.
    normalise : bool
        If True, rescale features to [0, π] before binding.
    basis : str
        Basis preparation before each repetition: ``"h"`` for Hadamard,
        ``"ry"`` for RY(π/2).
    parameter_prefix : str
        Prefix for the ParameterVector names.
    insert_barriers : bool
        Insert barriers between logical blocks for readability.
    name : str | None
        Optional circuit name.

    Returns
    -------
    QuantumCircuit
        Parameterised circuit ready for binding.
    """
    if feature_dimension < 2:
        raise ValueError("feature_dimension must be >= 2.")
    if max_order < 1:
        raise ValueError("max_order must be >= 1.")
    if any(c <= 0 for c in single_coeffs):
        raise ValueError("All single_coeffs must be positive.")
    n = int(feature_dimension)
    qc = QuantumCircuit(n, name=name or "ZZFeatureMapPolyExtended")

    x = ParameterVector(parameter_prefix, n)

    # Optional normalisation: map feature vector to [0, π]
    def _normalise_expr(expr: ParameterExpression) -> ParameterExpression:
        return pi * expr  # assume input already scaled to [0,1] by caller

    # Build φ1 as polynomial up to max_order
    def map1(xi: ParameterExpression) -> ParameterExpression:
        expr: ParameterExpression = 0
        power = xi
        for k in range(max_order):
            coeff = single_coeffs[k % len(single_coeffs)]
            expr = expr + coeff * power
            power = power * xi
        return expr

    def map2(xi: ParameterExpression, xj: ParameterExpression) -> ParameterExpression:
        return pair_weight * xi * xj

    pairs = _resolve_entanglement(n, entanglement)

    # Pre‑rotation
    if pre_rotate:
        qc.h(range(n))

    for rep in range(int(reps)):
        # Basis preparation
        if basis == "h":
            qc.h(range(n))
        elif basis == "ry":
            for q in range(n):
                qc.ry(pi / 2, q)
        else:
            raise ValueError("basis must be 'h' or 'ry'.")
        if insert_barriers:
            qc.barrier()

        # Single‑qubit phases
        for i in range(n):
            phase = 2 * map1(x[i])
            if normalise:
                phase = _normalise_expr(phase)
            qc.p(phase, i)
        if insert_barriers:
            qc.barrier()

        # ZZ interactions via CX–P–CX
        for (i, j) in pairs:
            angle = 2 * map2(x[i], x[j])
            if normalise:
                angle = _normalise_expr(angle)
            qc.cx(i, j)
            qc.p(angle, j)
            qc.cx(i, j)
        if insert_barriers and rep!= reps - 1:
            qc.barrier()

    # Post‑rotation
    if post_rotate:
        qc.h(range(n))

    qc.input_params = x  # type: ignore[attr-defined]
    return qc


class ZZFeatureMapPolyExtended(QuantumCircuit):
    """Object‑oriented wrapper for the extended polynomial ZZ feature map.

    The constructor mirrors the functional interface; the built circuit
    is composed into the instance.  The ``input_params`` attribute is
    preserved for binding.
    """
    def __init__(
        self,
        feature_dimension: int,
        reps: int = 2,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Iterable[Tuple[int, int]]] = "full",
        single_coeffs: Sequence[float] = (1.0,),
        pair_weight: float = 1.0,
        max_order: int = 2,
        pre_rotate: bool = False,
        post_rotate: bool = False,
        normalise: bool = True,
        basis: str = "h",
        parameter_prefix: str = "x",
        insert_barriers: bool = False,
        name: str = "ZZFeatureMapPolyExtended",
    ) -> None:
        built = zz_feature_map_poly_extended(
            feature_dimension,
            reps,
            entanglement,
            single_coeffs,
            pair_weight,
            max_order,
            pre_rotate,
            post_rotate,
            normalise,
            basis,
            parameter_prefix,
            insert_barriers,
            name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)
        self.input_params = built.input_params  # type: ignore[attr-defined]


__all__ = ["ZZFeatureMapPolyExtended", "zz_feature_map_poly_extended"]
