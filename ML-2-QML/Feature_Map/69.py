"""Controlled‑modified ZZ‑FeatureMap with symmetric pair‑phase and optional per‑pair weights."""
from __future__ import annotations

from math import pi
from typing import Callable, Sequence, Tuple, Union, List

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector, ParameterExpression


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _resolve_entanglement(
    num_qubits: int,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]],
) -> List[Tuple[int, int]]:
    """Return a list of two‑qubit pairs according to a simple entanglement spec.

    Supported specs:
      - ``"full"``: all‑to‑all pairs (i < j)
      - ``"linear"``: nearest neighbours (0,1), (1,2), …
      - ``"circular"``: linear plus wrap‑around (n-1,0) if n > 2
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
        if entanglement == "none":
            return []
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
# Controlled‑modified Polynomial Feature Map
# ---------------------------------------------------------------------------

def zz_feature_map_poly_controlled(
    feature_dimension: int,
    reps: int = 2,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
    single_coeffs: Sequence[float] = (1.0,),
    pair_weight: float = 1.0,
    pair_weights: Sequence[float] | None = None,
    basis: str = "h",  # "h" or "ry"
    normalize: bool = False,
    insert_barriers: bool = False,
    name: str | None = None,
) -> QuantumCircuit:
    """
    Controlled‑modified polynomial ZZ‑feature map.

    Parameters
    ----------
    feature_dimension : int
        Number of qubits / dimensionality of the input feature vector.
    reps : int, default 2
        Number of repetitions of the encoding block.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]]
        Entanglement topology. See :func:`_resolve_entanglement` for supported values.
    single_coeffs : Sequence[float], default (1.0,)
        Coefficients for the polynomial mapping of single‑qubit parameters.
    pair_weight : float, default 1.0
        Global weight applied to all pair interactions when ``pair_weights`` is None.
    pair_weights : Sequence[float] | None, default None
        Optional per‑pair weights. Must match the number of entanglement pairs.
    basis : str, default "h"
        Pre‑rotation basis: ``"h"`` for Hadamard, ``"ry"`` for RY(π/2).
    normalize : bool, default False
        If True, scale all angles by 2π to map raw data in [0,1] to [0,2π].
    insert_barriers : bool, default False
        Insert barriers between logical blocks for visual clarity.
    name : str | None, default None
        Optional circuit name.

    Returns
    -------
    QuantumCircuit
        Encoded feature map circuit.

    Notes
    -----
    * **Symmetric pair‑phase** – each ZZ interaction is implemented as a CX–P–CX
      sequence with the phase applied to *both* qubits (i and j).
    * **Per‑pair weighting** – allows fine‑grained control over interaction strength.
    * **Normalization** – useful when feeding data in the [0, 1] range.
    * **Entanglement flexibility** – supports full, linear, circular, or custom topologies.
    """
    if feature_dimension < 2:
        raise ValueError("feature_dimension must be >= 2.")
    n = int(feature_dimension)
    qc = QuantumCircuit(n, name=name or "ZZFeatureMapPolyControlled")

    x = ParameterVector("x", n)

    def map1(xi: ParameterExpression) -> ParameterExpression:
        expr: ParameterExpression = 0
        p = xi
        for c in single_coeffs:
            expr += c * p
            p *= xi
        return expr

    def map2(xi: ParameterExpression, xj: ParameterExpression, weight: float) -> ParameterExpression:
        return weight * xi * xj

    pairs = _resolve_entanglement(n, entanglement)

    # Validate per‑pair weights
    if pair_weights is None:
        pair_weights = [pair_weight] * len(pairs)
    else:
        if len(pair_weights)!= len(pairs):
            raise ValueError(
                f"pair_weights length ({len(pair_weights)}) does not match number "
                f"of entanglement pairs ({len(pairs)})."
            )
    pair_weights = list(pair_weights)

    angle_scale = 2 * pi if normalize else 1.0

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
            qc.p(2 * angle_scale * map1(x[i]), i)
        if insert_barriers:
            qc.barrier()

        # Symmetric ZZ interactions
        for idx, (i, j) in enumerate(pairs):
            angle = 2 * angle_scale * map2(x[i], x[j], pair_weights[idx])
            qc.cx(i, j)
            qc.p(angle, i)
            qc.p(angle, j)
            qc.cx(i, j)
        if insert_barriers and rep!= reps - 1:
            qc.barrier()

    qc.input_params = x  # type: ignore[attr-defined]
    return qc


class ZZFeatureMapPolyControlled(QuantumCircuit):
    """Class‑style wrapper for the controlled‑modified polynomial ZZ‑feature map."""
    def __init__(
        self,
        feature_dimension: int,
        reps: int = 2,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
        single_coeffs: Sequence[float] = (1.0,),
        pair_weight: float = 1.0,
        pair_weights: Sequence[float] | None = None,
        basis: str = "h",
        normalize: bool = False,
        insert_barriers: bool = False,
        name: str = "ZZFeatureMapPolyControlled",
    ) -> None:
        built = zz_feature_map_poly_controlled(
            feature_dimension,
            reps,
            entanglement,
            single_coeffs,
            pair_weight,
            pair_weights,
            basis,
            normalize,
            insert_barriers,
            name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)
        self.input_params = built.input_params  # type: ignore[attr-defined]


__all__ = ["ZZFeatureMapPolyControlled", "zz_feature_map_poly_controlled"]
