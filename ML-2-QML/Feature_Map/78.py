"""ZZFeatureMapRZZExtended: a richer RZZ‑based feature map with optional normalisation, higher‑order interactions and data‑dependent pre‑rotations."""
from __future__ import annotations

from math import pi
from typing import Callable, List, Sequence, Tuple

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterExpression, ParameterVector


# --------------------------------------------------------------------------- #
# Utility helpers
# --------------------------------------------------------------------------- #
def _resolve_entanglement(
    num_qubits: int,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]],
) -> List[Tuple[int, int]]:
    """Return a list of two‑qubit pairs according to a simple entanglement spec.

    Supported specs:
        * ``"full"`` – all‑to‑all pairs (i < j)
        * ``"linear"`` – nearest‑neighbor pairs
        * ``"circular"`` – linear + wrap‑around
        * explicit list of pairs or a callable that returns a sequence of (i, j)
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


# --------------------------------------------------------------------------- #
# Feature‑map construction
# --------------------------------------------------------------------------- #
def zz_feature_map_rzz_extended(
    feature_dimension: int,
    *,
    reps: int = 2,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
    data_map_func: Callable[[Sequence[ParameterExpression]], ParameterExpression] | None = None,
    parameter_prefix: str = "x",
    insert_barriers: bool = False,
    pair_scale: float = 1.0,
    higher_order_depth: int = 0,
    higher_order_scale: float = 0.5,
    pre_rotation: bool = False,
    pre_rotation_scale: float = 1.0,
    normalize: bool = False,
    name: str | None = None,
) -> QuantumCircuit:
    """
    Build an extended RZZ‑based feature map.

    Parameters
    ----------
    feature_dimension : int
        Number of classical input features (must be >= 2).
    reps : int, optional
        Number of repetition blocks (default 2).
    entanglement : str | Sequence[Tuple[int, int]] | Callable, optional
        Specification of qubit pairs to entangle. See ``_resolve_entanglement``.
    data_map_func : Callable[[Sequence[ParameterExpression]], ParameterExpression], optional
        Custom mapping from raw input parameters to rotation angles.
    parameter_prefix : str, optional
        Prefix used for the :class:`~qiskit.circuit.ParameterVector`.
    insert_barriers : bool, optional
        Whether to insert barriers between logical blocks.
    pair_scale : float, optional
        Global scaling factor for the pairwise RZZ angles.
    higher_order_depth : int, optional
        Number of additional entanglement layers that use ``higher_order_scale``.
    higher_order_scale : float, optional
        Scaling factor for the higher‑order entanglement angles.
    pre_rotation : bool, optional
        If True, a pre‑rotation layer (RZ) is applied before the main encoding.
    pre_rotation_scale : float, optional
        Scaling factor for the pre‑rotation angles.
    normalize : bool, optional
        If True, scale all input angles by ``1 / feature_dimension`` to keep them in a
        numerically stable range.
    name : str, optional
        Name of the resulting circuit.

    Returns
    -------
    QuantumCircuit
        The constructed feature‑map circuit.

    Notes
    -----
    * The circuit always consists of ``reps`` blocks of the form:

        1. (optional) Pre‑rotation RZ
        2. Hadamard on all qubits
        3. Phase (P) gates with angle ``2 * φ1(xi)``
        4. RZZ entanglers with angle ``2 * pair_scale * φ2(xi, xj)``
        5. (optional) Higher‑order RZZ layers with angle ``2 * higher_order_scale * φ2(xi, xj)``

    * ``data_map_func`` must accept a list of :class:`ParameterExpression` objects and
      return a single :class:`ParameterExpression`.  It is used to implement both
      single‑qubit and two‑qubit mappings.

    * The circuit exposes an ``input_params`` attribute containing the
      :class:`ParameterVector` used for binding.
    """
    if feature_dimension < 2:
        raise ValueError("feature_dimension must be >= 2.")
    if higher_order_depth < 0:
        raise ValueError("higher_order_depth must be >= 0.")
    if reps < 1:
        raise ValueError("reps must be >= 1.")

    n = int(feature_dimension)
    qc = QuantumCircuit(n, name=name or "ZZFeatureMapRZZExtended")

    # Parameter vector for the input features
    x = ParameterVector(parameter_prefix, n)

    # Define mapping functions
    if data_map_func is None:
        def map1(xi: ParameterExpression) -> ParameterExpression:
            return _default_map_1(xi)
        def map2(xi: ParameterExpression, xj: ParameterExpression) -> ParameterExpression:
            return _default_map_2(xi, xj)
    else:
        def map1(xi: ParameterExpression) -> ParameterExpression:
            return data_map_func([xi])
        def map2(xi: ParameterExpression, xj: ParameterExpression) -> ParameterExpression:
            return data_map_func([xi, xj])

    # Resolve entanglement pairs
    pairs = _resolve_entanglement(n, entanglement)

    # Optional normalisation factor
    norm_factor = 1.0 / n if normalize else 1.0

    for rep in range(int(reps)):
        if pre_rotation:
            for i in range(n):
                qc.rz(pre_rotation_scale * map1(x[i]), i)
        qc.h(range(n))
        if insert_barriers:
            qc.barrier()
        for i in range(n):
            qc.p(2 * norm_factor * map1(x[i]), i)
        if insert_barriers:
            qc.barrier()
        for (i, j) in pairs:
            qc.rzz(2 * pair_scale * norm_factor * map2(x[i], x[j]), i, j)
        # Higher‑order interaction layers
        for _ in range(higher_order_depth):
            for (i, j) in pairs:
                qc.rzz(2 * higher_order_scale * norm_factor * map2(x[i], x[j]), i, j)
        if insert_barriers and rep!= reps - 1:
            qc.barrier()

    qc.input_params = x  # type: ignore[attr-defined]
    return qc


class ZZFeatureMapRZZExtended(QuantumCircuit):
    """
    Object‑oriented wrapper for :func:`zz_feature_map_rzz_extended`.

    Parameters
    ----------
    feature_dimension : int
        Number of classical input features.
    reps : int, optional
        Number of repetition blocks.
    entanglement : str | Sequence[Tuple[int, int]] | Callable, optional
        Entanglement pattern.
    data_map_func : Callable[[Sequence[ParameterExpression]], ParameterExpression], optional
        Custom mapping function.
    parameter_prefix : str, optional
        Prefix for the :class:`ParameterVector`.
    insert_barriers : bool, optional
        Whether to insert barriers.
    pair_scale : float, optional
        Scaling for pairwise RZZ angles.
    higher_order_depth : int, optional
        Number of higher‑order entanglement layers.
    higher_order_scale : float, optional
        Scaling for higher‑order RZZ angles.
    pre_rotation : bool, optional
        Pre‑rotation layer flag.
    pre_rotation_scale : float, optional
        Scaling for pre‑rotation angles.
    normalize : bool, optional
        Normalise input angles.
    name : str, optional
        Circuit name.

    Notes
    -----
    The instance exposes an ``input_params`` attribute containing the
    :class:`ParameterVector` for easy parameter binding.
    """
    def __init__(
        self,
        feature_dimension: int,
        *,
        reps: int = 2,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
        data_map_func: Callable[[Sequence[ParameterExpression]], ParameterExpression] | None = None,
        parameter_prefix: str = "x",
        insert_barriers: bool = False,
        pair_scale: float = 1.0,
        higher_order_depth: int = 0,
        higher_order_scale: float = 0.5,
        pre_rotation: bool = False,
        pre_rotation_scale: float = 1.0,
        normalize: bool = False,
        name: str = "ZZFeatureMapRZZExtended",
    ) -> None:
        built = zz_feature_map_rzz_extended(
            feature_dimension,
            reps=reps,
            entanglement=entanglement,
            data_map_func=data_map_func,
            parameter_prefix=parameter_prefix,
            insert_barriers=insert_barriers,
            pair_scale=pair_scale,
            higher_order_depth=higher_order_depth,
            higher_order_scale=higher_order_scale,
            pre_rotation=pre_rotation,
            pre_rotation_scale=pre_rotation_scale,
            normalize=normalize,
            name=name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)
        self.input_params = built.input_params  # type: ignore[attr-defined]


__all__ = ["ZZFeatureMapRZZExtended", "zz_feature_map_rzz_extended"]
