


"""
Standalone, lightweight re-implementation of Qiskit's classic `ZZFeatureMap`
**plus variants** in a single-file module. This lets you A/B test feature-map
structures without importing the (deprecated) `qiskit.circuit.library.ZZFeatureMap`.

Included builders (class + function for each):

• `ZZFeatureMap` / `zz_feature_map(...)`
    - Canonical map: H → single-qubit phases → ZZ entanglers via CX–P–CX.
    - φ1(x)=x, φ2(x,y)=(π−x)(π−y) by default.

• `ZZFeatureMapRZZ` / `zz_feature_map_rzz(...)`
    - Same as canonical, but pair entanglers are native `rzz` rotations.
    - Optional `pair_scale` to amplify/attenuate pair phase.

• `ZZFeatureMapPoly` / `zz_feature_map_poly(...)`
    - Polynomial single-qubit phase and product pair phase.
      φ1(x)=∑_k c_k x^{k+1}; φ2(x,y)=w·x·y.
    - Choose basis prep: `basis='h'` (Hadamard) or `basis='ry'` (RY(π/2)).

All variants depend only on Qiskit core circuit objects (`QuantumCircuit`,
`ParameterVector`). Each circuit exposes a convenience attribute `input_params`
(a `ParameterVector`) to bind classical data easily.

Example
-------
>>> from zz_feature_map_with_variants import ZZFeatureMap, ZZFeatureMapRZZ, ZZFeatureMapPoly
>>> fm_base = ZZFeatureMap(3, reps=1, entanglement="linear")
>>> fm_rzz  = ZZFeatureMapRZZ(3, reps=1, entanglement="linear", pair_scale=0.5)
>>> fm_poly = ZZFeatureMapPoly(3, reps=1, single_coeffs=(1.0, -0.25), pair_weight=0.8, basis="ry")
"""
from __future__ import annotations

from math import pi
from typing import Callable, List, Sequence, Tuple

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector, ParameterExpression


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _resolve_entanglement(
    num_qubits: int,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]],
) -> List[Tuple[int, int]]:
    """Return a list of two-qubit pairs according to a simple entanglement spec.

    Supported specs:
      - "full": all-to-all pairs (i < j)
      - "linear": nearest neighbors (0,1), (1,2), ...
      - "circular": linear plus wrap-around (n-1,0) if n > 2
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
    """Default φ2(x, y) = (π − x)(π − y)."""
    return (pi - x) * (pi - y)


# ---------------------------------------------------------------------------
# Canonical ZZFeatureMap (CX–P–CX for ZZ)
# ---------------------------------------------------------------------------

def zz_feature_map(
    feature_dimension: int,
    reps: int = 2,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
    data_map_func: Callable[[Sequence[ParameterExpression]], ParameterExpression] | None = None,
    parameter_prefix: str = "x",
    insert_barriers: bool = False,
    name: str | None = None,
) -> QuantumCircuit:
    """Build the canonical ZZ-feature-map `QuantumCircuit`.

    Structure per repetition: H → P(2·φ1) on each qubit → ZZ entanglers via CX–P–CX
    with angle 2·φ2 on the target.
    """
    if feature_dimension < 2:
        raise ValueError("feature_dimension must be >= 2 for ZZFeatureMap.")

    n = int(feature_dimension)
    qc = QuantumCircuit(n, name=name or "ZZFeatureMap")

    x = ParameterVector(parameter_prefix, n)

    if data_map_func is None:
        map1 = _default_map_1
        map2 = _default_map_2
    else:
        def map1(xi: ParameterExpression) -> ParameterExpression:
            return data_map_func([xi])
        def map2(xi: ParameterExpression, xj: ParameterExpression) -> ParameterExpression:
            return data_map_func([xi, xj])

    pairs = _resolve_entanglement(n, entanglement)

    for rep in range(int(reps)):
        # Basis prep
        qc.h(range(n))
        if insert_barriers:
            qc.barrier()
        # Single-qubit phases
        for i in range(n):
            qc.p(2 * map1(x[i]), i)
        if insert_barriers:
            qc.barrier()
        # ZZ via CX–P–CX
        for (i, j) in pairs:
            angle_2 = 2 * map2(x[i], x[j])
            qc.cx(i, j)
            qc.p(angle_2, j)
            qc.cx(i, j)
        if insert_barriers and rep != reps - 1:
            qc.barrier()

    qc.input_params = x  # type: ignore[attr-defined]
    return qc


class ZZFeatureMap(QuantumCircuit):
    """Class-style wrapper for the canonical ZZFeatureMap."""
    def __init__(
        self,
        feature_dimension: int,
        reps: int = 2,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
        data_map_func: Callable[[Sequence[ParameterExpression]], ParameterExpression] | None = None,
        parameter_prefix: str = "x",
        insert_barriers: bool = False,
        name: str = "ZZFeatureMap",
    ) -> None:
        built = zz_feature_map(
            feature_dimension=feature_dimension,
            reps=reps,
            entanglement=entanglement,
            data_map_func=data_map_func,
            parameter_prefix=parameter_prefix,
            insert_barriers=insert_barriers,
            name=name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)
        self.input_params = built.input_params  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Variant 1: RZZ entanglers (native two-qubit rotation)
# ---------------------------------------------------------------------------

def zz_feature_map_rzz(
    feature_dimension: int,
    reps: int = 2,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
    data_map_func: Callable[[Sequence[ParameterExpression]], ParameterExpression] | None = None,
    parameter_prefix: str = "x",
    insert_barriers: bool = False,
    pair_scale: float = 1.0,
    name: str | None = None,
) -> QuantumCircuit:
    """ZZ feature map variant using native `rzz` for pair coupling.

    Pair angle is 2·pair_scale·φ2(xi, xj). Defaults to φ1(x)=x, φ2(x,y)=(π−x)(π−y).
    """
    if feature_dimension < 2:
        raise ValueError("feature_dimension must be >= 2.")
    n = int(feature_dimension)
    qc = QuantumCircuit(n, name=name or "ZZFeatureMapRZZ")

    x = ParameterVector(parameter_prefix, n)

    if data_map_func is None:
        def map1(xi: ParameterExpression) -> ParameterExpression:
            return xi
        def map2(xi: ParameterExpression, xj: ParameterExpression) -> ParameterExpression:
            return (pi - xi) * (pi - xj)
    else:
        def map1(xi: ParameterExpression) -> ParameterExpression:
            return data_map_func([xi])
        def map2(xi: ParameterExpression, xj: ParameterExpression) -> ParameterExpression:
            return data_map_func([xi, xj])

    pairs = _resolve_entanglement(n, entanglement)

    for rep in range(int(reps)):
        qc.h(range(n))
        if insert_barriers:
            qc.barrier()
        for i in range(n):
            qc.p(2 * map1(x[i]), i)
        if insert_barriers:
            qc.barrier()
        for (i, j) in pairs:
            qc.rzz(2 * pair_scale * map2(x[i], x[j]), i, j)
        if insert_barriers and rep != reps - 1:
            qc.barrier()

    qc.input_params = x  # type: ignore[attr-defined]
    return qc


class ZZFeatureMapRZZ(QuantumCircuit):
    """Class-style wrapper for the RZZ-entangler variant."""
    def __init__(
        self,
        feature_dimension: int,
        reps: int = 2,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
        data_map_func: Callable[[Sequence[ParameterExpression]], ParameterExpression] | None = None,
        parameter_prefix: str = "x",
        insert_barriers: bool = False,
        pair_scale: float = 1.0,
        name: str = "ZZFeatureMapRZZ",
    ) -> None:
        built = zz_feature_map_rzz(
            feature_dimension, reps, entanglement, data_map_func,
            parameter_prefix, insert_barriers, pair_scale, name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)
        self.input_params = built.input_params  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Variant 2: Polynomial single-qubit map + product pair map; H/RY basis
# ---------------------------------------------------------------------------

def zz_feature_map_poly(
    feature_dimension: int,
    reps: int = 2,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
    single_coeffs: Sequence[float] = (1.0,),
    pair_weight: float = 1.0,
    basis: str = "h",  # "h" or "ry"
    parameter_prefix: str = "x",
    insert_barriers: bool = False,
    name: str | None = None,
) -> QuantumCircuit:
    """ZZ feature map with polynomial φ1 and product φ2.

    φ1(x) = ∑_k single_coeffs[k] · x^{k+1}
    φ2(x,y) = pair_weight · x · y
    Basis prep can be Hadamards ("h") or RY(π/2) ("ry").
    """
    if feature_dimension < 2:
        raise ValueError("feature_dimension must be >= 2.")
    n = int(feature_dimension)
    qc = QuantumCircuit(n, name=name or "ZZFeatureMapPoly")

    x = ParameterVector(parameter_prefix, n)

    def map1(xi: ParameterExpression) -> ParameterExpression:
        expr: ParameterExpression = 0
        p = xi
        for c in single_coeffs:
            expr = expr + c * p
            p = p * xi  # next power
        return expr

    def map2(xi: ParameterExpression, xj: ParameterExpression) -> ParameterExpression:
        return pair_weight * xi * xj

    pairs = _resolve_entanglement(n, entanglement)

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

        # Single-qubit phases
        for i in range(n):
            qc.p(2 * map1(x[i]), i)
        if insert_barriers:
            qc.barrier()

        # ZZ via CX–P–CX
        for (i, j) in pairs:
            angle = 2 * map2(x[i], x[j])
            qc.cx(i, j)
            qc.p(angle, j)
            qc.cx(i, j)
        if insert_barriers and rep != reps - 1:
            qc.barrier()

    qc.input_params = x  # type: ignore[attr-defined]
    return qc


class ZZFeatureMapPoly(QuantumCircuit):
    """Class-style wrapper for the polynomial/basis variant."""
    def __init__(
        self,
        feature_dimension: int,
        reps: int = 2,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
        single_coeffs: Sequence[float] = (1.0,),
        pair_weight: float = 1.0,
        basis: str = "h",
        parameter_prefix: str = "x",
        insert_barriers: bool = False,
        name: str = "ZZFeatureMapPoly",
    ) -> None:
        built = zz_feature_map_poly(
            feature_dimension, reps, entanglement, single_coeffs, pair_weight,
            basis, parameter_prefix, insert_barriers, name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)
        self.input_params = built.input_params  # type: ignore[attr-defined]


__all__ = [
    # canonical
    "ZZFeatureMap",
    "zz_feature_map",
    # variants
    "ZZFeatureMapRZZ",
    "zz_feature_map_rzz",
    "ZZFeatureMapPoly",
    "zz_feature_map_poly",
]
