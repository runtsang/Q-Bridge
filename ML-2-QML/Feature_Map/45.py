"""ZZFeatureMapSharedParam builder (Hadamard + ZZ entanglement via CX–P–CX with shared parameters)."""

from __future__ import annotations

from math import pi
from typing import Callable, List, Sequence, Tuple

from sympy import sqrt

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterExpression, ParameterVector


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _resolve_entanglement(
    num_qubits: int,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]],
) -> List[Tuple[int, int]]:
    """Resolve the entanglement specification into an explicit list of qubit pairs.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]]
        Entanglement specification. Supported string values are ``"full"``, ``"linear"``, and ``"circular"``.
        A user can also provide an explicit list of pairs or a callable returning such a list.

    Returns
    -------
    List[Tuple[int, int]]
        List of valid qubit pairs.

    Raises
    ------
    ValueError
        If the specification is unknown or contains invalid pairs.
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


# ---------------------------------------------------------------------------
# Canonical ZZFeatureMap with shared pair parameters
# ---------------------------------------------------------------------------

def zz_feature_map_shared_param(
    feature_dimension: int,
    reps: int = 2,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
    data_map_func: Callable[[Sequence[ParameterExpression]], ParameterExpression] | None = None,
    parameter_prefix: str = "x",
    insert_barriers: bool = False,
    name: str | None = None,
    normalise: bool = False,
) -> QuantumCircuit:
    """Build a ZZ‑feature‑map with shared pair parameters.

    The circuit follows the canonical structure:
        H → P(2·φ1) on each qubit → ZZ entanglers via CX–P–CX
    but replaces the per‑pair phase with a single shared parameter θ_{ij} per
    qubit pair.  This reduces the number of parameters from *O(n²)* to *O(n + n²)*,
    where the *n* parameters correspond to single‑qubit rotations.

    Parameters
    ----------
    feature_dimension : int
        Number of classical features / qubits. Must be >= 2.
    reps : int, default 2
        Number of repetitions of the feature‑map block.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]], default "full"
        Entanglement specification. See ``_resolve_entanglement`` for details.
    data_map_func : Callable[[Sequence[ParameterExpression]], ParameterExpression] | None, default None
        Optional function that transforms the raw feature vector before encoding.
        Should accept a list of :class:`ParameterExpression` and return a single
        :class:`ParameterExpression`.
    parameter_prefix : str, default "x"
        Prefix for the parameter names of the single‑qubit angles.
    insert_barriers : bool, default False
        Insert barrier instructions after each block for visual clarity.
    name : str | None, default None
        Optional name for the resulting :class:`QuantumCircuit`.
    normalise : bool, default False
        If ``True`` the input vector is normalised to unit Euclidean norm before
        mapping.  This is useful for datasets where feature magnitudes vary
        widely.

    Returns
    -------
    QuantumCircuit
        The constructed feature‑map circuit.  Its ``input_params`` attribute
        contains the :class:`ParameterVector` for the single‑qubit angles.

    Raises
    ------
    ValueError
        If any argument is invalid (e.g., negative dimensions, unknown entanglement).
    """
    # --- Argument validation -------------------------------------------------
    if feature_dimension < 2:
        raise ValueError("feature_dimension must be >= 2 for ZZFeatureMapSharedParam.")
    if reps < 1:
        raise ValueError("reps must be >= 1.")
    if not isinstance(parameter_prefix, str):
        raise TypeError("parameter_prefix must be a string.")
    if not isinstance(normalise, bool):
        raise TypeError("normalise must be a bool.")
    if not isinstance(insert_barriers, bool):
        raise TypeError("insert_barriers must be a bool.")
    if data_map_func is not None and not callable(data_map_func):
        raise TypeError("data_map_func must be callable if supplied.")

    n = int(feature_dimension)
    qc = QuantumCircuit(n, name=name or "ZZFeatureMapSharedParam")

    # Single‑qubit parameters
    x = ParameterVector(parameter_prefix, n)

    # Resolve entanglement and build parameter vector for pair angles
    pairs = _resolve_entanglement(n, entanglement)
    if not pairs:
        raise ValueError("Entanglement specification yielded no pairs.")
    pair_params = ParameterVector(f"{parameter_prefix}_pair", len(pairs))
    pair_index = {pair: idx for idx, pair in enumerate(pairs)}

    # Data mapping helpers ----------------------------------------------------
    if normalise:
        norm_factor = sqrt(sum(x[i] ** 2 for i in range(n)))
        def map1(xi: ParameterExpression) -> ParameterExpression:
            base = xi / norm_factor
            return data_map_func([base]) if data_map_func else base
    else:
        def map1(xi: ParameterExpression) -> ParameterExpression:
            return data_map_func([xi]) if data_map_func else xi

    # -------------------------------------------------------------------------
    # Build the circuit
    # -------------------------------------------------------------------------
    for rep in range(int(reps)):
        # Basis preparation
        qc.h(range(n))
        if insert_barriers:
            qc.barrier()

        # Single‑qubit phases
        for i in range(n):
            qc.p(2 * map1(x[i]), i)

        if insert_barriers:
            qc.barrier()

        # ZZ entanglers with shared parameters
        for (i, j) in pairs:
            theta = pair_params[pair_index[(i, j)]]
            angle_2 = 2 * theta
            qc.cx(i, j)
            qc.p(angle_2, j)
            qc.cx(i, j)

        if insert_barriers and rep!= reps - 1:
            qc.barrier()

    qc.input_params = x  # type: ignore[attr-defined]
    return qc


class ZZFeatureMapSharedParam(QuantumCircuit):
    """Object‑oriented wrapper for :func:`zz_feature_map_shared_param`."""

    def __init__(
        self,
        feature_dimension: int,
        reps: int = 2,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
        data_map_func: Callable[[Sequence[ParameterExpression]], ParameterExpression] | None = None,
        parameter_prefix: str = "x",
        insert_barriers: bool = False,
        name: str = "ZZFeatureMapSharedParam",
        normalise: bool = False,
    ) -> None:
        built = zz_feature_map_shared_param(
            feature_dimension=feature_dimension,
            reps=reps,
            entanglement=entanglement,
            data_map_func=data_map_func,
            parameter_prefix=parameter_prefix,
            insert_barriers=insert_barriers,
            name=name,
            normalise=normalise,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)
        self.input_params = built.input_params  # type: ignore[attr-defined]


__all__ = ["ZZFeatureMapSharedParam", "zz_feature_map_shared_param"]
