"""Quantum feature map for a symmetric, normalised polynomial ZZ‑style map.
The class and function are defined in a single module, Qiskit‑compatible.
"""

from __future__ import annotations

from math import pi, sqrt
from typing import Callable, List, Sequence, Tuple

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterExpression, ParameterVector

# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #

def _resolve_entanglement(
    num_qubits: int,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]],
) -> List[Tuple[int, int]]:
    """Return a list of two‑qubit pairs according to a simple entanglement spec.

    Supported specs:
      - "full": all‑to‑all pairs (i < j)
      - "linear": nearest neighbours (0,1), (1,2), …
      - "circular": linear plus wrap‑around (n‑1,0) if n > 2
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

# --------------------------------------------------------------------------- #
# Feature Map
# --------------------------------------------------------------------------- #

def zz_feature_map_poly_controlled(
    feature_dimension: int,
    reps: int = 2,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
    single_coeffs: Sequence[float] = (1.0,),
    pair_weight: ParameterExpression | float = 1.0,
    normalise_interactions: bool = False,
    basis: str = "h",  # "h" or "ry"
    parameter_prefix: str = "x",
    insert_barriers: bool = False,
    name: str | None = None,
    scale_features: bool = False,
) -> QuantumCircuit:
    """
    Symmetric polynomial ZZ feature map with a shared pair‑weight.

    Parameters
    ----------
    feature_dimension : int
        Number of qubits / dimensionality of the input vector.
    reps : int, default 2
        Number of repetition blocks.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]]
        Specification of two‑qubit connections.
    single_coeffs : Sequence[float], default (1.0,)
        Coefficients for the single‑qubit polynomial φ1(x) = Σ_k c_k x^{k+1}.
    pair_weight : ParameterExpression | float, default 1.0
        Shared weight applied to every pair interaction. If a ParameterExpression is
        supplied, it can be bound later.
    normalise_interactions : bool, default False
        If True, divide the pair_weight by sqrt(number_of_pairs) to keep the total
        phase budget bounded.
    basis : str, default "h"
        Basis preparation: "h" for Hadamard, "ry" for RY(π/2).
    parameter_prefix : str, default "x"
        Prefix for the ParameterVector that holds the input features.
    insert_barriers : bool, default False
        Insert barriers between logical groups for visual clarity.
    name : str | None, default None
        Name of the resulting circuit; defaults to "ZZFeatureMapPolyControlled".
    scale_features : bool, default False
        If True, scale each input feature by 1/√n before feeding it into the map.

    Returns
    -------
    QuantumCircuit
        A parameterised circuit ready to bind classical data.

    Design Notes
    ------------
    * The pair‑weight is shared across all two‑qubit terms, enforcing a symmetric
      interaction structure.
    * Optional normalisation of the pair‑weight keeps the total phase budget
      from growing with the number of qubits.
    * Optional scaling of the input features keeps the magnitude of the single‑qubit
      phases bounded.
    * The single‑qubit polynomial map is unchanged from the seed, preserving
      expressivity for the first‑order terms.
    """
    if feature_dimension < 2:
        raise ValueError("feature_dimension must be >= 2.")
    if reps < 1:
        raise ValueError("reps must be >= 1.")
    if basis not in ("h", "ry"):
        raise ValueError("basis must be 'h' or 'ry'.")

    n = int(feature_dimension)
    qc = QuantumCircuit(n, name=name or "ZZFeatureMapPolyControlled")

    # Parameter vector for the input features
    x = ParameterVector(parameter_prefix, n)

    # Normalise pair_weight if requested
    pair_w = pair_weight
    if normalise_interactions:
        num_pairs = len(_resolve_entanglement(n, entanglement))
        # If pair_weight is a float, compute float division; if ParameterExpression,
        # the division returns a ParameterExpression.
        pair_w = pair_weight / sqrt(num_pairs)

    # Optional scaling of input features
    scaling_factor = 1.0
    if scale_features:
        scaling_factor = 1.0 / sqrt(n)

    def map1(xi: ParameterExpression) -> ParameterExpression:
        """Single‑qubit polynomial φ1(x)."""
        expr: ParameterExpression = 0
        power = xi * scaling_factor  # first power
        for c in single_coeffs:
            expr += c * power
            power *= xi * scaling_factor  # next power
        return expr

    def map2(xi: ParameterExpression, xj: ParameterExpression) -> ParameterExpression:
        """Pair interaction φ2(x, y)."""
        return pair_w * xi * xj

    pairs = _resolve_entanglement(n, entanglement)

    for rep in range(int(reps)):
        # Basis preparation
        if basis == "h":
            qc.h(range(n))
        else:  # basis == "ry"
            for q in range(n):
                qc.ry(pi / 2, q)

        if insert_barriers:
            qc.barrier()

        # Single‑qubit phases
        for i in range(n):
            qc.p(2 * map1(x[i]), i)

        if insert_barriers:
            qc.barrier()

        # ZZ interactions via CX–P–CX
        for (i, j) in pairs:
            angle = 2 * map2(x[i], x[j])
            qc.cx(i, j)
            qc.p(angle, j)
            qc.cx(i, j)

        if insert_barriers and rep!= reps - 1:
            qc.barrier()

    # Store the parameter vector for convenient binding
    qc.input_params = x  # type: ignore[attr-defined]
    return qc


class ZZFeatureMapPolyControlled(QuantumCircuit):
    """
    OO wrapper for the symmetric polynomial ZZ feature map.

    Parameters
    ----------
    Same as zz_feature_map_poly_controlled.

    The wrapper simply constructs the circuit via the helper function and
    exposes the resulting ParameterVector as ``input_params`` for easy data
    binding.
    """

    def __init__(
        self,
        feature_dimension: int,
        reps: int = 2,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
        single_coeffs: Sequence[float] = (1.0,),
        pair_weight: ParameterExpression | float = 1.0,
        normalise_interactions: bool = False,
        basis: str = "h",
        parameter_prefix: str = "x",
        insert_barriers: bool = False,
        name: str = "ZZFeatureMapPolyControlled",
        scale_features: bool = False,
    ) -> None:
        built = zz_feature_map_poly_controlled(
            feature_dimension,
            reps,
            entanglement,
            single_coeffs,
            pair_weight,
            normalise_interactions,
            basis,
            parameter_prefix,
            insert_barriers,
            name,
            scale_features,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)
        self.input_params = built.input_params  # type: ignore[attr-defined]


__all__ = ["ZZFeatureMapPolyControlled", "zz_feature_map_poly_controlled"]
