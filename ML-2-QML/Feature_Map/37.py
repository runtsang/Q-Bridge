"""
Quantum Feature Map module with controlled modification.
Implements a symmetrised ZZ-entanglement pattern and a shared‑parameter data map.
"""

from __future__ import annotations

import math
from typing import Callable, List, Sequence, Tuple

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector, ParameterExpression


# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #

def _resolve_entanglement(
    *,
    num_qubits: int,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]],
) -> List[Tuple[int, int]]:
    """
    Resolve an entanglement specification into a list of two‑qubit pairs.

    Supported specifications:
    - ``"full"``: all-to-all pairs (i < j)
    - ``"linear"``: nearest‑neighbour chain
    - ``"circular"``: linear plus wrap‑around
    - ``"full_sym"`` or ``"symmetric"``: all pairs with both directions (i,j) and (j,i)
    - explicit list of pairs ``[(0, 2), (1, 3)]``
    - callable ``f(num_qubits) -> sequence of (i, j)``

    Raises
    ------
    ValueError
        If the specification is unsupported or contains invalid pairs.
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
        if entanglement in ("full_sym", "symmetric"):
            base = [(i, j) for i in range(num_qubits) for j in range(i + 1, num_qubits)]
            return base + [(j, i) for (i, j) in base]
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


def _normalize_data(data: Sequence[float]) -> List[float]:
    """
    Scale data to the interval [0, π] for use with the feature map.

    Parameters
    ----------
    data : Sequence[float]
        Input feature vector.

    Returns
    -------
    List[float]
        Normalised data.
    """
    if not data:
        return []
    max_abs = max(abs(x) for x in data) or 1.0
    return [float(x) / max_abs * math.pi for x in data]


# --------------------------------------------------------------------------- #
# Canonical ZZ‑FeatureMap with Controlled Modification
# --------------------------------------------------------------------------- #

def zz_feature_map_controlled(
    feature_dimension: int,
    reps: int = 1,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
    data_map_func: Callable[[Sequence[ParameterExpression]], ParameterExpression] | None = None,
    parameter_prefix: str = "x",
    insert_barriers: bool = False,
    name: str | None = None,
    normalize: bool = False,
) -> QuantumCircuit:
    """
    Build a ZZ‑feature‑map with a shared‑parameter data map and a symmetrised entanglement pattern.

    The circuit structure per repetition is:
        H → P(2·φ1) on each qubit → ZZ entanglers via CX–P–CX
    where φ1 is a single‑qubit data function and φ2 is a shared‑parameter pair function.

    Parameters
    ----------
    feature_dimension : int
        Number of qubits / features. Must be >= 2.
    reps : int, default 1
        Number of repetitions of the basic block.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]]
        Entanglement specification. See :func:`_resolve_entanglement` for details.
    data_map_func : Callable[[Sequence[ParameterExpression]], ParameterExpression] | None, default None
        Custom mapping from data parameters to rotation angles.
        If None, default mappings are used:
        * φ1(x) = x
        * φ2(x, y) = (x + y) * z0
          where z0 is a shared ParameterVector of length 1.
    parameter_prefix : str, default "x"
        Prefix for the single‑qubit parameter vector.
    insert_barriers : bool, default False
        Whether to insert barriers between logical blocks.
    name : str | None, default None
        Optional circuit name. If None, defaults to ``"ZZFeatureMapControlled"``.
    normalize : bool, default False
        If True, data passed to :func:`bind_parameters` should be normalised to [0, π].
        The helper :func:`_normalize_data` is provided for convenience.

    Returns
    -------
    QuantumCircuit
        The constructed feature‑map circuit.
    """
    if feature_dimension < 2:
        raise ValueError("feature_dimension must be >= 2 for ZZFeatureMapControlled.")
    if reps < 1:
        raise ValueError("reps must be >= 1.")
    n = int(feature_dimension)
    qc = QuantumCircuit(n, name=name or "ZZFeatureMapControlled")

    # Parameter vectors
    x = ParameterVector(parameter_prefix, n)
    z = ParameterVector("z", 1)  # shared parameter for pair interactions

    # Default mapping functions
    if data_map_func is None:
        def map1(xi: ParameterExpression) -> ParameterExpression:
            return xi

        def map2(xi: ParameterExpression, xj: ParameterExpression) -> ParameterExpression:
            return (xi + xj) * z[0]
    else:
        def map1(xi: ParameterExpression) -> ParameterExpression:
            return data_map_func([xi])

        def map2(xi: ParameterExpression, xj: ParameterExpression) -> ParameterExpression:
            return data_map_func([xi, xj])

    # Resolve entanglement pairs
    pairs = _resolve_entanglement(num_qubits=n, entanglement=entanglement)

    for rep in range(int(reps)):
        # Basis preparation
        qc.h(range(n))
        if insert_barriers:
            qc.barrier()

        # Single‑qubit phase rotations
        for i in range(n):
            qc.p(2 * map1(x[i]), i)

        if insert_barriers:
            qc.barrier()

        # ZZ interactions via CX–P–CX
        for (i, j) in pairs:
            angle_2 = 2 * map2(x[i], x[j])
            qc.cx(i, j)
            qc.p(angle_2, j)
            qc.cx(i, j)

        if insert_barriers and rep!= reps - 1:
            qc.barrier()

    # Expose input parameters for binding
    qc.input_params = list(x) + list(z)  # type: ignore[attr-defined]
    return qc


class ZZFeatureMapControlled(QuantumCircuit):
    """
    Object‑oriented wrapper for the controlled‑modification ZZ‑feature‑map.

    Parameters
    ----------
    feature_dimension : int
        Number of qubits / features.
    reps : int, default 1
        Number of repetitions.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]]
        Entanglement specification.
    data_map_func : Callable[[Sequence[ParameterExpression]], ParameterExpression] | None, default None
        Custom data‑to‑angle mapping.
    parameter_prefix : str, default "x"
        Prefix for the single‑qubit parameters.
    insert_barriers : bool, default False
        Insert barriers between logical blocks.
    normalize : bool, default False
        If True, data should be normalised to [0, π] before binding.
    name : str, default "ZZFeatureMapControlled"
        Circuit name.
    """

    def __init__(
        self,
        feature_dimension: int,
        reps: int = 1,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
        data_map_func: Callable[[Sequence[ParameterExpression]], ParameterExpression] | None = None,
        parameter_prefix: str = "x",
        insert_barriers: bool = False,
        normalize: bool = False,
        name: str = "ZZFeatureMapControlled",
    ) -> None:
        built = zz_feature_map_controlled(
            feature_dimension=feature_dimension,
            reps=reps,
            entanglement=entanglement,
            data_map_func=data_map_func,
            parameter_prefix=parameter_prefix,
            insert_barriers=insert_barriers,
            name=name,
            normalize=normalize,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)
        self.input_params = built.input_params  # type: ignore[attr-defined]


__all__ = ["ZZFeatureMapControlled", "zz_feature_map_controlled"]
