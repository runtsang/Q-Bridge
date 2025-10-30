"""ZZFeatureMapRZZControlled variant with shared entanglement parameters and normalisation."""
from __future__ import annotations

from math import pi
from typing import Callable, List, Sequence, Tuple

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterExpression, ParameterVector

# --------------------------------------------------------------------------- #
# Helper utilities
# --------------------------------------------------------------------------- #
def _resolve_entanglement(
    num_qubits: int,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]],
) -> List[Tuple[int, int]]:
    """Resolve entanglement specification into a list of two‑qubit pairs.

    Supported specs:
      - "full": all‑to‑all (i < j)
      - "linear": nearest‑neighbors only.
      - "circular": linear with wrap‑around.
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
# Default mapping functions
# --------------------------------------------------------------------------- #
def _default_map_1(x: ParameterExpression) -> ParameterExpression:
    """Default φ1(x) = sin(x) for single‑qubit rotations."""
    return (pi / 2) * x  # scaled to keep angles in [0,π] for typical data in [0,1]


def _default_map_2(x: ParameterExpression, y: ParameterExpression) -> ParameterExpression:
    """Default φ2(x, y) = cos(x) * cos(y) for pairwise couplings."""
    return (pi / 2) * x * y


# --------------------------------------------------------------------------- #
# Variant 2: Controlled‑modification RZZ entanglers
# --------------------------------------------------------------------------- #
def zz_feature_map_rzz_controlled(
    feature_dimension: int,
    reps: int = 2,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
    data_map_func: Callable[[Sequence[ParameterExpression]], ParameterExpression] | None = None,
    parameter_prefix: str = "x",
    insert_barriers: bool = False,
    pair_scale: float = 1.0,
    normalise: bool = False,
    name: str | None = None,
) -> QuantumCircuit:
    """ZZ feature map variant with controlled‑modification of the entanglement.

    Parameters
    ----------
    feature_dimension : int
        Number of classical features / qubits. Must be >= 2.
    reps : int, default 2
        Number of repeat layers.
    entanglement : str | Sequence | Callable, default "full"
        Entanglement pattern. See :func:`_resolve_entanglement`.
    data_map_func : Callable[[Sequence[ParameterExpression]], ParameterExpression] | None, default None
        Custom mapping from classical data to rotation angles. If None, defaults to
        sin(x) for single‑qubit rotations and cos(x)*cos(y) for pairwise couplings.
    parameter_prefix : str, default "x"
        Prefix for the ParameterVector names.
    insert_barriers : bool, default False
        If True, inserts barriers between layers to aid circuit visualisation.
    pair_scale : float, default 1.0
        Global scaling factor for all RZZ angles. Must be non‑zero.
    normalise : bool, default False
        If True, rescales ``pair_scale`` by 1/√feature_dimension to keep the overall
        entanglement strength comparable across different qubit counts.
    name : str | None, default None
        Optional circuit name.

    Returns
    -------
    QuantumCircuit
        The constructed feature‑map circuit with parameter vector ``input_params``.
    """
    if feature_dimension < 2:
        raise ValueError("feature_dimension must be >= 2.")
    if pair_scale == 0.0:
        raise ValueError("pair_scale must be non‑zero to avoid degenerate RZZ angles.")
    n = int(feature_dimension)
    qc = QuantumCircuit(n, name=name or "ZZFeatureMapRZZControlled")

    # Parameter vector for all qubit‑wise data
    x = ParameterVector(parameter_prefix, n)

    # Optional normalisation of the pair scale
    scale = pair_scale / (n ** 0.5) if normalise else pair_scale

    # Define mapping functions
    if data_map_func is None:
        map1 = _default_map_1
        map2 = _default_map_2
    else:
        # Wrap the user function to accept the expected number of arguments
        def map1(xi: ParameterExpression) -> ParameterExpression:
            return data_map_func([xi])

        def map2(xi: ParameterExpression, xj: ParameterExpression) -> ParameterExpression:
            return data_map_func([xi, xj])

    pairs = _resolve_entanglement(n, entanglement)

    for rep in range(int(reps)):
        qc.h(range(n))
        if insert_barriers:
            qc.barrier()
        for i in range(n):
            qc.p(2 * map1(x[i]), i)
        if insert_barriers:
            qc.barrier()
        for (i, j) in pairs:
            qc.rzz(2 * scale * map2(x[i], x[j]), i, j)
        if insert_barriers and rep!= reps - 1:
            qc.barrier()

    qc.input_params = x  # type: ignore[attr-defined]
    return qc


class ZZFeatureMapRZZControlled(QuantumCircuit):
    """Class‑style wrapper for the controlled‑modification RZZ‑entangler variant."""

    def __init__(
        self,
        feature_dimension: int,
        reps: int = 2,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
        data_map_func: Callable[[Sequence[ParameterExpression]], ParameterExpression] | None = None,
        parameter_prefix: str = "x",
        insert_barriers: bool = False,
        pair_scale: float = 1.0,
        normalise: bool = False,
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
            normalise,
            name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)
        self.input_params = built.input_params  # type: ignore[attr-defined]


__all__ = ["ZZFeatureMapRZZControlled", "zz_feature_map_rzz_controlled"]
