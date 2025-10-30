"""ZZFeatureMapRZZControlled variant with symmetrised pairwise entanglement and shared scaling."""
from __future__ import annotations

from math import pi
from typing import Callable, List, Sequence, Tuple

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterExpression, ParameterVector

# --------------------------------------------------------------------------- #
# Helper utilities
# --------------------------------------------------------------------------- #

def _resolve_entanglement(
    *,
    num_qubits: int,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]],
) -> List[Tuple[int, int]]:
    """Return a spec‑wise list of two‑qubit pairs.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]]
        * ``"full"`` – all‑to‑all pairs (i < j)
        * ``"linear"`` – nearest neighbours (0,1), (1,2), …
        * ``"circular"`` – linear plus wrap‑around (n‑1,0)
        * explicit list of pairs
        * callable f(n) → sequence of (i, j)

    Returns
    -------
    List[Tuple[int, int]]
        Validated list of qubit pairs.

    Raises
    ------
    ValueError
        If an unknown string spec is supplied or a pair is invalid.
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
# Feature‑map construction
# --------------------------------------------------------------------------- #

def zz_feature_map_rzz_controlled(
    feature_dimension: int,
    reps: int = 2,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
    data_map_func: Callable[[Sequence[ParameterExpression]], ParameterExpression] | None = None,
    parameter_prefix: str = "x",
    insert_barriers: bool = False,
    pair_scale: float = 1.0,
    global_scale: float = 1.0,
    normalise_data: bool = False,
    name: str | None = None,
) -> QuantumCircuit:
    """Build a ZZ‑feature map with controlled‑modification scaling.

    The circuit follows the original RZZ‑based entanglement pattern but introduces
    two additional knobs:

    * ``global_scale`` – a single parameter that multiplies all pair‑wise RZZ
      angles, effectively synchronising the strength of every entanglement.
    * ``normalise_data`` – when ``True`` each classical feature is scaled to the
      interval [0, π] by multiplying with ``π/2``.  This is useful when the
      input data is known to lie in [0, 1].

    Parameters
    ----------
    feature_dimension : int
        Number of classical features / qubits.
    reps : int, default 2
        Number of repetitions of the feature‑map block.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]]
        Specification of two‑qubit entanglement pairs.
    data_map_func : Callable[[Sequence[ParameterExpression]], ParameterExpression] | None
        Optional custom mapping from raw parameters to the arguments of the
        single‑ and two‑qubit rotations.  When ``None`` the default mappings
        ``φ1(x)=x`` and ``φ2(x,y)=(π−x)(π−y)`` are used.
    parameter_prefix : str, default "x"
        Prefix for the parameters in the :class:`~qiskit.circuit.ParameterVector`.
    insert_barriers : bool, default False
        Whether to insert barriers between logical layers for easier visual
        inspection.
    pair_scale : float, default 1.0
        Scaling applied to each pair‑wise RZZ angle before the global scaling.
    global_scale : float, default 1.0
        Global factor multiplying **all** pair‑wise RZZ angles.
    normalise_data : bool, default False
        If ``True`` each input parameter is multiplied by ``π/2`` before being
        fed into the mapping functions.
    name : str | None, default None
        Optional name for the resulting :class:`~qiskit.circuit.QuantumCircuit`.

    Returns
    -------
    QuantumCircuit
        A parameterised circuit ready for data binding.

    Raises
    ------
    ValueError
        If any numeric input is out of the expected range.
    """
    if feature_dimension < 2:
        raise ValueError("feature_dimension must be >= 2.")
    if reps < 1:
        raise ValueError("reps must be >= 1.")
    if pair_scale < 0:
        raise ValueError("pair_scale must be non‑negative.")
    if global_scale < 0:
        raise ValueError("global_scale must be non‑negative.")

    n = int(feature_dimension)
    qc = QuantumCircuit(n, name=name or "ZZFeatureMapRZZControlled")

    # Parameter vector
    x = ParameterVector(parameter_prefix, n)

    # Normalisation factor
    norm_factor = pi / 2 if normalise_data else 1.0

    # Default mapping functions
    if data_map_func is None:

        def map1(xi: ParameterExpression) -> ParameterExpression:
            """Default φ1(x) = x."""
            return xi * norm_factor

        def map2(xi: ParameterExpression, xj: ParameterExpression) -> ParameterExpression:
            """Default φ2(x,y) = (π − x)(π − y)."""
            return (pi - xi * norm_factor) * (pi - xj * norm_factor)

    else:

        def map1(xi: ParameterExpression) -> ParameterExpression:
            return data_map_func([xi * norm_factor])

        def map2(xi: ParameterExpression, xj: ParameterExpression) -> ParameterExpression:
            return data_map_func([xi * norm_factor, xj * norm_factor])

    # Resolve entanglement pairs
    pairs = _resolve_entanglement(num_qubits=n, entanglement=entanglement)

    for rep in range(int(reps)):
        qc.h(range(n))
        if insert_barriers:
            qc.barrier()
        for i in range(n):
            qc.p(2 * map1(x[i]), i)
        if insert_barriers:
            qc.barrier()
        for (i, j) in pairs:
            angle = 2 * pair_scale * global_scale * map2(x[i], x[j])
            qc.rzz(angle, i, j)
        if insert_barriers and rep!= reps - 1:
            qc.barrier()

    qc.input_params = x  # type: ignore[attr-defined]
    return qc


class ZZFeatureMapRZZControlled(QuantumCircuit):
    """Class‑style wrapper for :func:`zz_feature_map_rzz_controlled`."""

    def __init__(
        self,
        feature_dimension: int,
        reps: int = 2,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
        data_map_func: Callable[[Sequence[ParameterExpression]], ParameterExpression] | None = None,
        parameter_prefix: str = "x",
        insert_barriers: bool = False,
        pair_scale: float = 1.0,
        global_scale: float = 1.0,
        normalise_data: bool = False,
        name: str = "ZZFeatureMapRZZControlled",
    ) -> None:
        built = zz_feature_map_rzz_controlled(
            feature_dimension,
            reps,
            entanglement,
            data_map_func,
            parameter_prefix,
            insert_barriers,
            pair_scale,
            global_scale,
            normalise_data,
            name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)
        self.input_params = built.input_params  # type: ignore[attr-defined]


__all__ = ["ZZFeatureMapRZZControlled", "zz_feature_map_rzz_controlled"]
