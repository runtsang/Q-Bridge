"""Extended ZZFeatureMap builder (Hadamard + ZZ + optional higher‑order interactions)."""

from __future__ import annotations

from math import pi
from typing import Callable, List, Sequence, Tuple, Union, Optional

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
      - "full": all‑to‑all pairs (i < j)
      - "linear": nearest neighbors (0,1), (1,2),...
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

    pairs = [(int(i), int(j)) for (i, j) in entanglement]  # type: ignore[arg-type]
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
# Extended ZZFeatureMap (CX–P–CX for ZZ + optional 3‑qubit ZZ)
# ---------------------------------------------------------------------------

def extended_zz_feature_map(
    feature_dimension: int,
    reps: int = 2,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
    data_map_func: Callable[[Sequence[ParameterExpression]], ParameterExpression] | None = None,
    parameter_prefix: str = "x",
    insert_barriers: bool = False,
    normalize: bool = False,
    pre_rotation: bool = False,
    post_rotation: bool = False,
    include_higher_order: bool = False,
    name: str | None = None,
) -> QuantumCircuit:
    """
    Build an extended ZZ‑feature‑map with optional higher‑order couplings.

    Parameters
    ----------
    feature_dimension : int
        Number of classical features (must be >= 2).
    reps : int, default 2
        Number of repetitions of the feature‑map block.
    entanglement : str | Sequence[Tuple[int, int]] | Callable
        Specification of two‑qubit entanglement pairs.
    data_map_func : Callable[[Sequence[ParameterExpression]], ParameterExpression] | None
        Function mapping a list of raw parameters to the phase used in the P gate.
        If None, the default pairwise and triplet mappings are used.
    parameter_prefix : str, default "x"
        Prefix for the generated ParameterVector.
    insert_barriers : bool, default False
        Insert barriers between logical blocks for readability.
    normalize : bool, default False
        If True, scale the raw feature values to the interval [0, π] before mapping.
    pre_rotation : bool, default False
        Add an optional Hadamard layer before the main feature‑map block.
    post_rotation : bool, default False
        Add an optional Hadamard layer after the main feature‑map block.
    include_higher_order : bool, default False
        Include 3‑qubit ZZ interactions using a CX–P–CX–CX sequence.
    name : str | None, default None
        Name of the resulting QuantumCircuit.

    Returns
    -------
    QuantumCircuit
        The constructed extended feature‑map circuit.

    Notes
    -----
    * The circuit depth grows linearly with ``reps`` and with the number of entangled pairs.
    * When ``include_higher_order`` is True, each 3‑qubit set (i, j, k) receives an additional
      XX‑ZZ‑style phase implemented as CX(i,j) → CX(i,k) → P → CX(i,k) → CX(i,j).
    * Parameters are exposed via ``qc.input_params`` for easy binding.

    Examples
    --------
    >>> from qiskit.circuit import ParameterVector
    >>> qc = extended_zz_feature_map(3, reps=1)
    >>> qc.input_params
    ParameterVector('x', 3)
    """
    if feature_dimension < 2:
        raise ValueError("feature_dimension must be >= 2 for ExtendedZZFeatureMap.")

    n = int(feature_dimension)
    qc = QuantumCircuit(n, name=name or "ExtendedZZFeatureMap")

    # Normalise raw data if requested
    raw_params = ParameterVector(parameter_prefix, n)
    if normalize:
        # Scale to [0, π] by multiplying by π
        params = [pi * rp for rp in raw_params]
    else:
        params = raw_params

    # Determine mapping functions
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

    for rep in range(int(reps)):
        # Optional pre‑rotation
        if pre_rotation:
            qc.h(range(n))
            if insert_barriers:
                qc.barrier()

        # Basis preparation
        qc.h(range(n))
        if insert_barriers:
            qc.barrier()

        # Single‑qubit phases
        for i in range(n):
            qc.p(2 * map1(params[i]), i)

        # Two‑qubit ZZ via CX–P–CX
        for (i, j) in pairs:
            angle = 2 * map2(params[i], params[j])
            qc.cx(i, j)
            qc.p(angle, j)
            qc.cx(i, j)

        # Optional higher‑order ZZ (three‑qubit) interactions
        if include_higher_order:
            for i in range(n - 2):
                for j in range(i + 1, n - 1):
                    for k in range(j + 1, n):
                        angle3 = 2 * map3(params[i], params[j], params[k])
                        qc.cx(i, j)
                        qc.cx(i, k)
                        qc.p(angle3, k)
                        qc.cx(i, k)
                        qc.cx(i, j)

        # Optional post‑rotation
        if post_rotation:
            qc.h(range(n))
            if insert_barriers:
                qc.barrier()

        if insert_barriers and rep!= reps - 1:
            qc.barrier()

    qc.input_params = raw_params  # type: ignore[attr-defined]
    return qc


class ExtendedZZFeatureMap(QuantumCircuit):
    """Object‑oriented wrapper for the extended ZZ‑feature‑map.

    The class accepts the same arguments as :func:`extended_zz_feature_map` and
    exposes the underlying circuit via ``self``.  It also attaches the
    ``input_params`` attribute for convenient parameter binding.

    Example
    -------
    >>> from qiskit.circuit import ParameterVector
    >>> x = ParameterVector('x', 4)
    >>> map = ExtendedZZFeatureMap(4, reps=1, include_higher_order=True)
    >>> map.bind_parameters({x[i]: 0.5 for i in range(4)})
    """
    def __init__(
        self,
        feature_dimension: int,
        reps: int = 2,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
        data_map_func: Callable[[Sequence[ParameterExpression]], ParameterExpression] | None = None,
        parameter_prefix: str = "x",
        insert_barriers: bool = False,
        normalize: bool = False,
        pre_rotation: bool = False,
        post_rotation: bool = False,
        include_higher_order: bool = False,
        name: str = "ExtendedZZFeatureMap",
    ) -> None:
        built = extended_zz_feature_map(
            feature_dimension=feature_dimension,
            reps=reps,
            entanglement=entanglement,
            data_map_func=data_map_func,
            parameter_prefix=parameter_prefix,
            insert_barriers=insert_barriers,
            normalize=normalize,
            pre_rotation=pre_rotation,
            post_rotation=post_rotation,
            include_higher_order=include_higher_order,
            name=name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)
        self.input_params = built.input_params  # type: ignore[attr-defined]


__all__ = ["ExtendedZZFeatureMap", "extended_zz_feature_map"]
