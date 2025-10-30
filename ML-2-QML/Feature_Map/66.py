from __future__ import annotations

from math import pi, prod
from itertools import combinations
from typing import Callable, Iterable, List, Sequence, Tuple, Optional

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector, ParameterExpression


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _resolve_entanglement(
    num_qubits: int,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Iterable[Tuple[int, int]]],
) -> List[Tuple[int, int]]:
    """
    Resolve a user‑supplied entanglement specification into a list of qubit pairs.

    Supported specs:
    - ``"full"``: all‑to‑all pairs (i < j)
    - ``"linear"``: nearest‑neighbour pairs (0,1), (1,2), …
    - ``"circular"``: linear plus wrap‑around (n‑1,0) if n > 2
    - explicit list of tuples like [(0, 2), (1, 3)]
    - callable: f(num_qubits) → iterable of (i, j)

    Raises
    ------
    ValueError
        If an unknown string is supplied or if a pair is invalid.
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


def _default_map_1(x: ParameterExpression) -> ParameterExpression:
    """Default φ₁(x) = x."""
    return x


def _default_map_N(xs: Sequence[ParameterExpression]) -> ParameterExpression:
    """Default φₙ(x₁,…,xₙ) = ∏ₖ (π − xₖ)."""
    return prod(pi - xi for xi in xs)


def _higher_order_combinations(n: int, order: int) -> List[Tuple[int,...]]:
    """Generate all unique qubit tuples of a given order."""
    if order < 2 or order > n:
        raise ValueError(f"interaction_order must satisfy 2 ≤ order ≤ n; got order={order}, n={n}")
    return list(combinations(range(n), order))


# ---------------------------------------------------------------------------
# Extended ZZFeatureMap (CX–P–CX for ZZ, enriched)
# ---------------------------------------------------------------------------

def extended_zz_feature_map(
    feature_dimension: int,
    reps: int = 2,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Iterable[Tuple[int, int]]] = "full",
    interaction_order: int = 2,
    data_map_func: Optional[Callable[[Sequence[ParameterExpression]], ParameterExpression]] = None,
    normalise: bool = False,
    data_scaling: float = 1.0,
    pre_rotation_func: Optional[Callable[[ParameterVector], List[ParameterExpression]]] = None,
    post_rotation_func: Optional[Callable[[ParameterVector], List[ParameterExpression]]] = None,
    parameter_prefix: str = "x",
    insert_barriers: bool = False,
    name: Optional[str] = None,
) -> QuantumCircuit:
    """
    Build an extended ZZ‑feature‑map circuit.

    The circuit follows the canonical structure:
        H → P(2·φ₁) → pairwise ZZ via CX–P–CX
    but adds optional:
    - higher‑order ZZ interactions (order ≥ 2)
    - data scaling
    - pre‑ and post‑rotation hooks
    - normalisation flag (currently a no‑op with a warning)

    Parameters
    ----------
    feature_dimension
        Number of classical features / qubits.
    reps
        Number of repetitions of the feature‑map block.
    entanglement
        Specification of pairwise entanglement.
    interaction_order
        Order of ZZ interactions to apply (≥2). 2 = pairwise, 3 = triple, etc.
    data_map_func
        Optional mapping from a list of feature parameters to a single angle.
        If None, default φ₁ or φₙ is used.
    normalise
        If True, a warning is issued that normalisation is not performed automatically.
    data_scaling
        A scalar applied to all feature parameters before mapping.
    pre_rotation_func
        Function returning a list of angles for P‑rotations applied before each repetition.
    post_rotation_func
        Function returning a list of angles for P‑rotations applied after each repetition.
    parameter_prefix
        Prefix for the ParameterVector names.
    insert_barriers
        Insert barriers between blocks for visual clarity.
    name
        Name of the resulting QuantumCircuit.

    Returns
    -------
    QuantumCircuit
        The constructed feature‑map circuit with an ``input_params`` attribute.
    """
    if feature_dimension < 2:
        raise ValueError("feature_dimension must be >= 2 for ZZFeatureMap.")
    if reps < 1:
        raise ValueError("reps must be >= 1.")
    if interaction_order < 2:
        raise ValueError("interaction_order must be >= 2.")
    if interaction_order > feature_dimension:
        raise ValueError("interaction_order cannot exceed feature_dimension.")
    if normalise:
        # Normalisation would require symbolic sqrt, which is not supported in ParameterExpression.
        # The user should normalise the data externally.
        print("Warning: normalise=True is a no‑op; please normalise your data before encoding.")

    n = int(feature_dimension)
    qc = QuantumCircuit(n, name=name or "ExtendedZZFeatureMap")

    # Parameters for the raw features
    raw_params = ParameterVector(parameter_prefix, n)
    # Apply data scaling
    scaled_params = [data_scaling * p for p in raw_params]

    # Map functions
    if data_map_func is None:
        map1 = _default_map_1
        mapN = _default_map_N
    else:
        def map1(xi: ParameterExpression) -> ParameterExpression:
            return data_map_func([xi])
        def mapN(xs: Sequence[ParameterExpression]) -> ParameterExpression:
            return data_map_func(list(xs))

    pairs = _resolve_entanglement(n, entanglement)
    higher_order_combos = _higher_order_combinations(n, interaction_order)

    for rep in range(int(reps)):
        # Pre‑rotation (if any)
        if pre_rotation_func is not None:
            pre_angles = pre_rotation_func(raw_params)
            if len(pre_angles)!= n:
                raise ValueError("pre_rotation_func must return a list of length n.")
            for i, ang in enumerate(pre_angles):
                qc.p(ang, i)

        # Basis preparation
        qc.h(range(n))
        if insert_barriers:
            qc.barrier()

        # Single‑qubit phases
        for i in range(n):
            qc.p(2 * map1(scaled_params[i]), i)

        if insert_barriers:
            qc.barrier()

        # Pairwise ZZ via CX–P–CX
        for (i, j) in pairs:
            angle_2 = 2 * mapN([scaled_params[i], scaled_params[j]])
            qc.cx(i, j)
            qc.p(angle_2, j)
            qc.cx(i, j)

        # Higher‑order ZZ interactions
        if interaction_order > 2:
            for combo in higher_order_combos:
                # Chain CXs from first to last qubit in combo
                for k in range(len(combo) - 1):
                    qc.cx(combo[k], combo[k + 1])
                angle = 2 * mapN([scaled_params[idx] for idx in combo])
                qc.p(angle, combo[-1])
                # Reverse chain
                for k in reversed(range(len(combo) - 1)):
                    qc.cx(combo[k], combo[k + 1])

        # Post‑rotation (if any)
        if post_rotation_func is not None:
            post_angles = post_rotation_func(raw_params)
            if len(post_angles)!= n:
                raise ValueError("post_rotation_func must return a list of length n.")
            for i, ang in enumerate(post_angles):
                qc.p(ang, i)

        if insert_barriers and rep!= reps - 1:
            qc.barrier()

    qc.input_params = raw_params  # type: ignore[attr-defined]
    return qc


class ExtendedZZFeatureMap(QuantumCircuit):
    """Class‑style wrapper for the extended ZZ‑feature‑map.

    Parameters
    ----------
    feature_dimension
        Number of qubits / classical features.
    reps
        Number of repetitions.
    entanglement
        Specification of pairwise entanglement (see :func:`_resolve_entanglement`).
    interaction_order
        Order of ZZ interactions (>1).
    data_map_func
        Optional mapping function from feature list to angle.
    normalise
        No‑op; see :func:`extended_zz_feature_map` for details.
    data_scaling
        Scalar applied to all features.
    pre_rotation_func
        Optional pre‑rotation hook.
    post_rotation_func
        Optional post‑rotation hook.
    parameter_prefix
        Prefix for parameters.
    insert_barriers
        Insert barriers between blocks.
    name
        Circuit name.

    Attributes
    ----------
    input_params
        ParameterVector of the raw feature parameters.
    """
    def __init__(
        self,
        feature_dimension: int,
        reps: int = 2,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Iterable[Tuple[int, int]]] = "full",
        interaction_order: int = 2,
        data_map_func: Optional[Callable[[Sequence[ParameterExpression]], ParameterExpression]] = None,
        normalise: bool = False,
        data_scaling: float = 1.0,
        pre_rotation_func: Optional[Callable[[ParameterVector], List[ParameterExpression]]] = None,
        post_rotation_func: Optional[Callable[[ParameterVector], List[ParameterExpression]]] = None,
        parameter_prefix: str = "x",
        insert_barriers: bool = False,
        name: str = "ExtendedZZFeatureMap",
    ) -> None:
        built = extended_zz_feature_map(
            feature_dimension=feature_dimension,
            reps=reps,
            entanglement=entanglement,
            interaction_order=interaction_order,
            data_map_func=data_map_func,
            normalise=normalise,
            data_scaling=data_scaling,
            pre_rotation_func=pre_rotation_func,
            post_rotation_func=post_rotation_func,
            parameter_prefix=parameter_prefix,
            insert_barriers=insert_barriers,
            name=name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)
        self.input_params = built.input_params  # type: ignore[attr-defined]


__all__ = ["ExtendedZZFeatureMap", "extended_zz_feature_map"]
