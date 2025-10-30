"""Extended Polynomial ZZFeatureMap with higher‑order interactions and data normalization."""
from __future__ import annotations

from math import pi
from typing import Callable, List, Sequence, Tuple, Union

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
    ],
) -> List[Tuple[int, int]]:
    """Return a list of two‑qubit pairs based on a simple entanglement spec.

    Supported specs:
        * ``full``: all‑to‑all pairs (i < j)
        * ``linear``: nearest‑neighbour pairs (i‑i+1)
        * ``circular``: linear plus wrap‑around
        * explicit list of pairs
        * callable returning a list of pairs
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
    for (i, j) in pairs:
        if i == j:
            raise ValueError("Entanglement pairs must connect distinct qubits.")
        if not (0 <= i < num_qubits and 0 <= j < num_qubits):
            raise ValueError(f"Entanglement pair {(i, j)} out of range for n={num_qubits}.")
    return pairs


def _resolve_triple_pairs(num_qubits: int) -> List[Tuple[int, int, int]]:
    """Return all unique triples of qubits (i < j < k)."""
    if num_qubits < 3:
        return []
    return [
        (i, j, k)
        for i in range(num_qubits)
        for j in range(i + 1, num_qubits)
        for k in range(j + 1, num_qubits)
    ]


# ---------------------------------------------------------------------------
# Feature map
# ---------------------------------------------------------------------------

def zz_feature_map_poly_extended(
    feature_dimension: int,
    reps: int = 2,
    entanglement: Union[
        str,
        Sequence[Tuple[int, int]],
        Callable[[int], Sequence[Tuple[int, int]]],
    ] = "full",
    single_coeffs: Sequence[float] = (1.0,),
    pair_weight: float = 1.0,
    triple_weight: float = 0.0,
    basis: str = "h",
    parameter_prefix: str = "x",
    insert_barriers: bool = False,
    data_scaling: bool = False,
    data_scale_factor: float = 1.0,
    name: Union[str, None] = None,
) -> QuantumCircuit:
    """Return a ZZ feature map with optional 3‑body interactions and data scaling.

    Parameters
    ----------
    feature_dimension : int
        Number of classical features / qubits. Must be >= 1.
    reps : int, default 2
        Number of repetitions of the encoding block.
    entanglement : str, Sequence or Callable
        Specification of two‑qubit entanglement pairs.
    single_coeffs : Sequence[float], default (1.0,)
        Coefficients for the polynomial of degree len(single_coeffs) on each qubit.
    pair_weight : float, default 1.0
        Weight of the pairwise ZZ interaction.
    triple_weight : float, default 0.0
        Weight of the triple‑body ZZ interaction. Ignored if 0.
    basis : str, default "h"
        Basis preparation: "h" for Hadamard, "ry" for RY(pi/2).
    parameter_prefix : str, default "x"
        Prefix for the parameter vector.
    insert_barriers : bool, default False
        Whether to insert barriers between layers.
    data_scaling : bool, default False
        If True, scales each feature by ``data_scale_factor`` before encoding.
    data_scale_factor : float, default 1.0
        Multiplicative factor applied when ``data_scaling`` is True.
    name : str or None, default None
        Name of the resulting circuit.

    Returns
    -------
    QuantumCircuit
        A parameterised circuit ready for binding with a feature vector.
    """
    if feature_dimension < 1:
        raise ValueError("feature_dimension must be >= 1.")
    n = int(feature_dimension)
    if pair_weight!= 0.0 and n < 2:
        raise ValueError("At least 2 qubits are required for pairwise interactions.")
    if triple_weight!= 0.0 and n < 3:
        raise ValueError("At least 3 qubits are required for triple interactions.")
    if data_scaling and data_scale_factor <= 0:
        raise ValueError("data_scale_factor must be positive when data_scaling is True.")

    qc = QuantumCircuit(n, name=name or "ZZFeatureMapPolyExtended")
    params = ParameterVector(parameter_prefix, n)

    scale = data_scale_factor if data_scaling else 1.0

    def map1(xi: ParameterExpression) -> ParameterExpression:
        """Polynomial on a single qubit."""
        expr: ParameterExpression = 0
        p = xi
        for c in single_coeffs:
            expr += c * p
            p *= xi
        return scale * expr

    def map2(xi: ParameterExpression, xj: ParameterExpression) -> ParameterExpression:
        """Pairwise product map."""
        return scale ** 2 * pair_weight * xi * xj

    def map3(
        xi: ParameterExpression, xj: ParameterExpression, xk: ParameterExpression
    ) -> ParameterExpression:
        """Triple‑body product map."""
        return scale ** 3 * triple_weight * xi * xj * xk

    pairs = _resolve_entanglement(n, entanglement)
    triples = _resolve_triple_pairs(n)

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
            qc.p(2 * map1(params[i]), i)

        if insert_barriers:
            qc.barrier()

        # Pairwise ZZ
        for (i, j) in pairs:
            angle = 2 * map2(params[i], params[j])
            qc.cx(i, j)
            qc.p(angle, j)
            qc.cx(i, j)

        # Triple‑body ZZ (if enabled)
        if triple_weight!= 0.0:
            for (i, j, k) in triples:
                angle = 2 * map3(params[i], params[j], params[k])
                # Implement a controlled‑controlled‑phase using CX‑CX‑P‑CX‑CX
                qc.cx(i, j)
                qc.cx(j, k)
                qc.p(angle, k)
                qc.cx(j, k)
                qc.cx(i, j)

        if insert_barriers and rep!= reps - 1:
            qc.barrier()

    qc.input_params = params  # type: ignore[attr-defined]
    return qc


class ZZFeatureMapPolyExtended(QuantumCircuit):
    """QuantumCircuit subclass for the extended ZZ feature map."""

    def __init__(
        self,
        feature_dimension: int,
        reps: int = 2,
        entanglement: Union[
            str,
            Sequence[Tuple[int, int]],
            Callable[[int], Sequence[Tuple[int, int]]],
        ] = "full",
        single_coeffs: Sequence[float] = (1.0,),
        pair_weight: float = 1.0,
        triple_weight: float = 0.0,
        basis: str = "h",
        parameter_prefix: str = "x",
        insert_barriers: bool = False,
        data_scaling: bool = False,
        data_scale_factor: float = 1.0,
        name: str = "ZZFeatureMapPolyExtended",
    ) -> None:
        built = zz_feature_map_poly_extended(
            feature_dimension,
            reps,
            entanglement,
            single_coeffs,
            pair_weight,
            triple_weight,
            basis,
            parameter_prefix,
            insert_barriers,
            data_scaling,
            data_scale_factor,
            name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)
        self.input_params = built.input_params  # type: ignore[attr-defined]


__all__ = ["ZZFeatureMapPolyExtended", "zz_feature_map_poly_extended"]
