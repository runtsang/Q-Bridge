"""Extended ZZFeatureMap with higher‑order interactions and optional pre/post rotations."""
from __future__ import annotations

from math import pi
from typing import Callable, List, Sequence, Tuple, Union

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterExpression, ParameterVector

# --------------------------------------------------------------------------- #
# Utility helpers
# --------------------------------------------------------------------------- #

def _resolve_entanglement(
    num_qubits: int,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]],
) -> List[Tuple[int, int]]:
    """
    Resolve a pairwise entanglement specification into a list of qubit pairs.
    Supported specs:
      - ``"full"``   : all-to-all pairs (i < j)
      - ``"linear"`` : nearest neighbours (0,1), (1,2), …
      - ``"circular"`` : linear plus wrap‑around (n‑1,0) if n > 2
      - explicit list of tuples
      - callable ``f(n) -> sequence of (i, j)``
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


def _resolve_triple_entanglement(
    num_qubits: int,
    entanglement: str | Sequence[Tuple[int, int, int]] | Callable[[int], Sequence[Tuple[int, int, int]]],
) -> List[Tuple[int, int, int]]:
    """
    Resolve a triple‑qubit entanglement specification into a list of triples.
    Supported specs:
      - ``"full"``   : all triples (i < j < k)
      - ``"linear"`` : consecutive triples (0,1,2), (1,2,3), …
      - ``"circular"`` : linear plus wrap‑around (n‑2,n‑1,0), (n‑1,0,1)
      - explicit list of tuples
      - callable ``f(n) -> sequence of (i, j, k)``
    """
    if isinstance(entanglement, str):
        if entanglement == "full":
            return [(i, j, k)
                    for i in range(num_qubits)
                    for j in range(i + 1, num_qubits)
                    for k in range(j + 1, num_qubits)]
        if entanglement == "linear":
            return [(i, i + 1, i + 2) for i in range(num_qubits - 2)]
        if entanglement == "circular":
            triples = [(i, i + 1, i + 2) for i in range(num_qubits - 2)]
            if num_qubits > 2:
                triples.append((num_qubits - 2, num_qubits - 1, 0))
                triples.append((num_qubits - 1, 0, 1))
            return triples
        raise ValueError(f"Unknown triple entanglement string: {entanglement!r}")

    if callable(entanglement):
        triples = list(entanglement(num_qubits))
        return [(int(i), int(j), int(k)) for (i, j, k) in triples]

    # sequence of triples
    triples = [(int(i), int(j), int(k)) for (i, j, k) in entanglement]  # type: ignore[arg-type]
    for (i, j, k) in triples:
        if len({i, j, k}) < 3:
            raise ValueError("Triple entanglement must involve three distinct qubits.")
        if not (0 <= i < num_qubits and 0 <= j < num_qubits and 0 <= k < num_qubits):
            raise ValueError(f"Triple entanglement {(i, j, k)} out of range for n={num_qubits}.")
    return triples


# --------------------------------------------------------------------------- #
# Default mapping functions
# --------------------------------------------------------------------------- #

def _default_map_1(x: ParameterExpression) -> ParameterExpression:
    """Default single‑qubit map: φ₁(x) = x."""
    return x


def _default_map_2(x: ParameterExpression, y: ParameterExpression) -> ParameterExpression:
    """Default two‑qubit map: φ₂(x, y) = (π − x)(π − y)."""
    return (pi - x) * (pi - y)


def _default_map_3(x: ParameterExpression, y: ParameterExpression, z: ParameterExpression) -> ParameterExpression:
    """Default three‑qubit map: φ₃(x, y, z) = (π − x)(π − y)(π − z)."""
    return (pi - x) * (pi - y) * (pi - z)


# --------------------------------------------------------------------------- #
# Core builder
# --------------------------------------------------------------------------- #

def extended_zz_feature_map(
    feature_dimension: int,
    reps: int = 2,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
    interaction_order: int = 2,
    data_map_func: Callable[[Sequence[ParameterExpression]], ParameterExpression] | None = None,
    pre_rotation: Callable[[Sequence[ParameterExpression]], ParameterVector] | None = None,
    post_rotation: Callable[[Sequence[ParameterExpression]], ParameterVector] | None = None,
    normalize: bool = False,
    parameter_prefix: str = "x",
    insert_barriers: bool = False,
    name: str | None = None,
) -> QuantumCircuit:
    """
    Build an extended ZZ‑feature‑map circuit.

    Parameters
    ----------
    feature_dimension : int
        Number of classical features (must be >= interaction_order).
    reps : int, default 2
        How many times the basic encoding block is repeated.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]]
        Specification of two‑qubit entanglement pairs.
    interaction_order : int, default 2
        2 for pairwise ZZ, 3 for three‑qubit ZZ interactions.  >3 not supported.
    data_map_func : Callable[[Sequence[ParameterExpression]], ParameterExpression], optional
        User‑supplied function that maps a list of parameters to a rotation angle.
        If None, built‑in defaults are used.
    pre_rotation : Callable[[Sequence[ParameterExpression]], ParameterVector], optional
        Function that returns a ParameterVector of length *n* to apply
        additional single‑qubit rotations before the main encoding block.
    post_rotation : Callable[[Sequence[ParameterExpression]], ParameterVector], optional
        Function that returns a ParameterVector of length *n* to apply
        additional single‑qubit rotations after the main encoding block.
    normalize : bool, default False
        If True, each input feature is scaled to the range [0, π] by multiplying by π.
    parameter_prefix : str, default "x"
        Prefix for the ParameterVector names.
    insert_barriers : bool, default False
        Whether to insert barriers between logical layers.
    name : str | None, default None
        Optional name for the resulting circuit.

    Returns
    -------
    QuantumCircuit
        A parameterised circuit ready for binding with a classical feature vector.
    """
    if interaction_order not in (2, 3):
        raise ValueError("interaction_order must be 2 (pairwise) or 3 (triple).")

    if feature_dimension < interaction_order:
        raise ValueError(f"feature_dimension must be >= {interaction_order} for interaction_order={interaction_order}.")

    n = int(feature_dimension)
    qc = QuantumCircuit(n, name=name or "ExtendedZZFeatureMap")

    # Parameter vector for raw features
    x = ParameterVector(parameter_prefix, n)

    # Normalisation: scale each feature to [0, π] if requested
    x_scaled = [pi * xi if normalize else xi for xi in x]

    # Resolve entanglement pairs/triples
    if interaction_order == 2:
        pairs = _resolve_entanglement(n, entanglement)
    else:  # interaction_order == 3
        pairs = _resolve_triple_entanglement(n, entanglement)  # type: ignore[assignment]

    # Mapping functions
    if data_map_func is None:
        map1 = _default_map_1
        map2 = _default_map_2
        map3 = _default_map_3
    else:
        # Wrap the user function to accept the appropriate number of arguments
        def map1(xi: ParameterExpression) -> ParameterExpression:
            return data_map_func([xi])

        def map2(xi: ParameterExpression, xj: ParameterExpression) -> ParameterExpression:
            return data_map_func([xi, xj])

        def map3(xi: ParameterExpression, xj: ParameterExpression, xk: ParameterExpression) -> ParameterExpression:
            return data_map_func([xi, xj, xk])

    # Pre‑rotation vector
    pre_vec: ParameterVector | None = None
    if pre_rotation is not None:
        pre_vec = pre_rotation(x_scaled)
        if len(pre_vec)!= n:
            raise ValueError("pre_rotation must return a ParameterVector of length equal to feature_dimension.")

    # Post‑rotation vector
    post_vec: ParameterVector | None = None
    if post_rotation is not None:
        post_vec = post_rotation(x_scaled)
        if len(post_vec)!= n:
            raise ValueError("post_rotation must return a ParameterVector of length equal to feature_dimension.")

    for rep in range(int(reps)):
        # Basis preparation
        qc.h(range(n))
        if insert_barriers:
            qc.barrier()

        # Optional pre‑rotations
        if pre_vec is not None:
            for i in range(n):
                qc.p(2 * pre_vec[i], i)

        # Single‑qubit phase rotations
        for i in range(n):
            qc.p(2 * map1(x_scaled[i]), i)

        # Entangling layers
        if interaction_order == 2:
            for (i, j) in pairs:  # type: ignore[assignment]
                angle = 2 * map2(x_scaled[i], x_scaled[j])
                qc.cx(i, j)
                qc.p(angle, j)
                qc.cx(i, j)
        else:  # interaction_order == 3
            for (i, j, k) in pairs:  # type: ignore[assignment]
                angle = 2 * map3(x_scaled[i], x_scaled[j], x_scaled[k])
                # Decomposition of a three‑qubit controlled‑phase
                qc.cx(i, j)
                qc.cx(j, k)
                qc.p(angle, k)
                qc.cx(j, k)
                qc.cx(i, j)

        if insert_barriers and rep!= reps - 1:
            qc.barrier()

    # Optional post‑rotations
    if post_vec is not None:
        for i in range(n):
            qc.p(2 * post_vec[i], i)

    qc.input_params = x  # type: ignore[attr-defined]
    return qc


# --------------------------------------------------------------------------- #
# Class‑style wrapper
# --------------------------------------------------------------------------- #

class ExtendedZZFeatureMap(QuantumCircuit):
    """
    Class‑style wrapper for the extended ZZ‑feature‑map.

    Parameters are identical to :func:`extended_zz_feature_map`.
    """

    def __init__(
        self,
        feature_dimension: int,
        reps: int = 2,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
        interaction_order: int = 2,
        data_map_func: Callable[[Sequence[ParameterExpression]], ParameterExpression] | None = None,
        pre_rotation: Callable[[Sequence[ParameterExpression]], ParameterVector] | None = None,
        post_rotation: Callable[[Sequence[ParameterExpression]], ParameterVector] | None = None,
        normalize: bool = False,
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
            pre_rotation=pre_rotation,
            post_rotation=post_rotation,
            normalize=normalize,
            parameter_prefix=parameter_prefix,
            insert_barriers=insert_barriers,
            name=name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)
        self.input_params = built.input_params  # type: ignore[attr-defined]


__all__ = ["ExtendedZZFeatureMap", "extended_zz_feature_map"]
