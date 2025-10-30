from __future__ import annotations

from math import pi
from typing import Callable, List, Sequence, Tuple
import itertools

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterExpression, ParameterVector


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _resolve_entanglement(
    num_qubits: int,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]],
) -> List[Tuple[int, int]]:
    """Return a list of two‑qubit pairs according to a simple entanglement spec.

    Supported specs:
      - "full": all‑to‑all pairs (i < j)
      - "linear": nearest neighbours (0,1), (1,2),...
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


def _default_map_1(x: ParameterExpression) -> ParameterExpression:
    """Default φ1(x) = x."""
    return x


def _default_map_2(x: ParameterExpression, y: ParameterExpression) -> ParameterExpression:
    """Default φ2(x, y) = (π − x)(π − y)."""
    return (pi - x) * (pi - y)


def _default_map_3(x: ParameterExpression, y: ParameterExpression, z: ParameterExpression) -> ParameterExpression:
    """Default φ3(x, y, z) = (π − x)(π − y)(π − z)."""
    return (pi - x) * (pi - y) * (pi - z)


# ---------------------------------------------------------------------------
# Canonical ZZFeatureMap (CX‑P‑CX for ZZ) – extended
# ---------------------------------------------------------------------------

def zz_feature_map(
    feature_dimension: int,
    reps: int = 2,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
    data_map_func: Callable[[Sequence[ParameterExpression]], ParameterExpression] | None = None,
    parameter_prefix: str = "x",
    insert_barriers: bool = False,
    interaction_order: int = 2,
    pre_rotation: Callable[[ParameterVector], Sequence[ParameterExpression]] | None = None,
    post_rotation: Callable[[ParameterVector], Sequence[ParameterExpression]] | None = None,
    normalize: bool = False,
    name: str | None = None,
) -> QuantumCircuit:
    """Build an extended ZZ‑feature‑map.

    The circuit follows the canonical Hadamard → single‑qubit phase → ZZ entanglement
    pattern, but offers:

    * **Interaction order** – 2 (pairwise) or 3 (three‑body) ZZ terms.
    * **Pre‑ and post‑rotations** – arbitrary P‑rotations applied before or after the core entanglement.
    * **Normalization** – multiply all angles by 2π to map raw data to a full period.
    * **Custom data mapping** – a user‑supplied function that receives a list of
      ParameterExpressions and returns the desired rotation angle.

    Parameters
    ----------
    feature_dimension : int
        Number of classical features / qubits. Must be >= interaction_order.
    reps : int, default 2
        Number of repetitions of the core pattern.
    entanglement : str | Sequence[Tuple[int, int]] | Callable
        Specification of two‑qubit pairs for ZZ entanglement.
    data_map_func : Callable[[Sequence[ParameterExpression]], ParameterExpression], optional
        Maps a list of features to an angle. If omitted, the defaults above are used.
    parameter_prefix : str, default "x"
        Prefix for the ParameterVector names.
    insert_barriers : bool, default False
        Insert barriers between logical blocks for visual clarity.
    interaction_order : int, default 2
        2 for pairwise ZZ, 3 for three‑body ZZ. Other values raise an error.
    pre_rotation : Callable[[ParameterVector], Sequence[ParameterExpression]], optional
        Function returning a list of angles to apply before the Hadamard layer.
    post_rotation : Callable[[ParameterVector], Sequence[ParameterExpression]], optional
        Function returning a list of angles to apply after the entanglement.
    normalize : bool, default False
        If True, multiply all angles by 2π.
    name : str, optional
        Circuit name.

    Returns
    -------
    QuantumCircuit
        The constructed feature‑map circuit with an ``input_params`` attribute.

    Raises
    ------
    ValueError
        If ``feature_dimension`` < ``interaction_order`` or if ``interaction_order`` is unsupported.
    """
    if feature_dimension < 2:
        raise ValueError("feature_dimension must be >= 2 for ZZFeatureMap.")
    if interaction_order not in (2, 3):
        raise ValueError("interaction_order must be 2 or 3.")
    if feature_dimension < interaction_order:
        raise ValueError(f"feature_dimension ({feature_dimension}) must be >= interaction_order ({interaction_order}).")

    n = int(feature_dimension)
    qc = QuantumCircuit(n, name=name or "ZZFeatureMap")

    x = ParameterVector(parameter_prefix, n)

    # Map functions
    if data_map_func is None:
        map1 = _default_map_1
        map2 = _default_map_2
        map3 = _default_map_3
    else:
        def map1(xi: ParameterExpression) -> ParameterExpression:
            return data_map_func([xi])

        def map2(xi: ParameterExpression, xj: ParameterExpression) -> ParameterExpression:
            return data_map_func([xi, xj])

        def map3(xi: ParameterExpression, xj: ParameterExpression, xk: ParameterExpression) -> ParameterExpression:
            return data_map_func([xi, xj, xk])

    pairs = _resolve_entanglement(n, entanglement)
    triples = list(itertools.combinations(range(n), 3)) if interaction_order == 3 else []

    factor = 2 * pi if normalize else 1

    for rep in range(int(reps)):
        # Pre‑rotation
        if pre_rotation is not None:
            pre_angles = pre_rotation(x)
            if len(pre_angles)!= n:
                raise ValueError("pre_rotation must return a list of length equal to the number of qubits.")
            for i in range(n):
                qc.p(pre_angles[i], i)

        # Basis preparation
        qc.h(range(n))
        if insert_barriers:
            qc.barrier()

        # Single‑qubit phases
        for i in range(n):
            qc.p(2 * factor * map1(x[i]), i)

        # Pairwise ZZ entanglement
        for (i, j) in pairs:
            angle_2 = 2 * factor * map2(x[i], x[j])
            qc.cx(i, j)
            qc.p(angle_2, j)
            qc.cx(i, j)

        # Three‑body ZZ entanglement (optional)
        if interaction_order == 3:
            for (i, j, k) in triples:
                angle_3 = 2 * factor * map3(x[i], x[j], x[k])
                qc.cx(i, j)
                qc.cx(j, k)
                qc.p(angle_3, k)
                qc.cx(j, k)
                qc.cx(i, j)

        # Post‑rotation
        if post_rotation is not None:
            post_angles = post_rotation(x)
            if len(post_angles)!= n:
                raise ValueError("post_rotation must return a list of length equal to the number of qubits.")
            for i in range(n):
                qc.p(post_angles[i], i)

        if insert_barriers and rep!= reps - 1:
            qc.barrier()

    qc.input_params = x  # type: ignore[attr-defined]
    return qc


class ZZFeatureMap(QuantumCircuit):
    """Class‑style wrapper for the extended ZZ‑feature‑map.

    Parameters
    ----------
    Same as :func:`zz_feature_map`.

    Notes
    -----
    The ``input_params`` attribute is preserved for easy parameter binding.
    """
    def __init__(
        self,
        feature_dimension: int,
        reps: int = 2,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
        data_map_func: Callable[[Sequence[ParameterExpression]], ParameterExpression] | None = None,
        parameter_prefix: str = "x",
        insert_barriers: bool = False,
        interaction_order: int = 2,
        pre_rotation: Callable[[ParameterVector], Sequence[ParameterExpression]] | None = None,
        post_rotation: Callable[[ParameterVector], Sequence[ParameterExpression]] | None = None,
        normalize: bool = False,
        name: str = "ZZFeatureMap",
    ) -> None:
        built = zz_feature_map(
            feature_dimension=feature_dimension,
            reps=reps,
            entanglement=entanglement,
            data_map_func=data_map_func,
            parameter_prefix=parameter_prefix,
            insert_barriers=insert_barriers,
            interaction_order=interaction_order,
            pre_rotation=pre_rotation,
            post_rotation=post_rotation,
            normalize=normalize,
            name=name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)
        self.input_params = built.input_params  # type: ignore[attr-defined]


__all__ = ["ZZFeatureMap", "zz_feature_map"]
