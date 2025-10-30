"""ZZFeatureMapRZZControlled variant with improved entanglement symmetry and optional per‑feature phase‑shift re‑parameterisation."""
from __future__ import annotations

from math import pi, sqrt
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
    """
    Resolve a user‑provided entanglement specification into a list of two‑qubit pairs.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]]
        Entanglement specification. Supported strings:
        - ``"full"``   : all‑to‑all pairs (i < j)
        - ``"linear"`` : nearest‑neighbour pairs (0,1), (1,2), …
        - ``"circular"`` : linear + wrap‑around (n-1,0) if n > 2
        - explicit list of pairs ``[(i, j), …]``
        - callable ``f(num_qubits) -> sequence of (i, j)``

    Returns
    -------
    List[Tuple[int, int]]
        List of two‑qubit indices to entangle.

    Raises
    ------
    ValueError
        If an unsupported specification is provided or a pair is invalid.
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

    # Sequence of pairs
    pairs = [(int(i), int(j)) for (i, j) in entanglement]  # type: ignore[arg-type]
    for (i, j) in pairs:
        if i == j:
            raise ValueError("Entanglement pairs must connect distinct qubits.")
        if not (0 <= i < num_qubits and 0 <= j < num_qubits):
            raise ValueError(f"Entanglement pair {(i, j)} out of range for n={num_qubits}.")
    return pairs


# --------------------------------------------------------------------------- #
# Feature‑map implementation
# --------------------------------------------------------------------------- #
def zz_feature_map_rzz_controlled(
    feature_dimension: int,
    reps: int = 2,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
    data_map_func: Callable[[Sequence[ParameterExpression]], ParameterExpression] | None = None,
    parameter_prefix: str = "x",
    insert_barriers: bool = False,
    base_pair_scale: float = 1.0,
    phase_shift_func: Callable[[ParameterExpression], ParameterExpression] | None = None,
    name: str | None = None,
) -> QuantumCircuit:
    """
    Controlled‑modification variant of the RZZ‑entangler feature map.

    The circuit symmetrises two‑qubit couplings by using a single RZZ gate per pair and a
    shared scaling factor that is inversely proportional to the square root of the
    number of qubits. This reduces depth while preserving expressive power.
    Additionally, a per‑feature phase‑shift re‑parameterisation can be supplied
    to make the map invariant to global phase shifts of the input data.

    Parameters
    ----------
    feature_dimension : int
        Number of classical features (must be >= 2).
    reps : int, optional
        Number of repetitions of the feature‑map pattern.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]], optional
        Entanglement specification.
    data_map_func : Callable[[Sequence[ParameterExpression]], ParameterExpression] | None, optional
        Function mapping a list of parameters to a single parameter. If None,
        defaults to ``φ1(x) = x`` and ``φ2(x, y) = (π − x)(π − y)``.
    parameter_prefix : str, optional
        Prefix for the ParameterVector names.
    insert_barriers : bool, optional
        Whether to insert barriers between layers for readability.
    base_pair_scale : float, optional
        Base scaling factor for pair couplings before normalisation.
    phase_shift_func : Callable[[ParameterExpression], ParameterExpression] | None, optional
        Function mapping each input parameter to a phase‑shifted value. If None,
        defaults to the identity.
    name : str | None, optional
        Name of the resulting QuantumCircuit.

    Returns
    -------
    QuantumCircuit
        The constructed feature‑map circuit with ``input_params`` attribute.

    Raises
    ------
    ValueError
        If ``feature_dimension`` < 2.
    """
    if feature_dimension < 2:
        raise ValueError("feature_dimension must be >= 2.")
    n = int(feature_dimension)

    qc = QuantumCircuit(n, name=name or "ZZFeatureMapRZZControlled")

    x = ParameterVector(parameter_prefix, n)

    # Default data mapping functions
    if data_map_func is None:
        def map1(xi: ParameterExpression) -> ParameterExpression:
            return xi

        def map2(xi: ParameterExpression, xj: ParameterExpression) -> ParameterExpression:
            return (pi - xi) * (pi - xj)
    else:
        def map1(xi: ParameterExpression) -> ParameterExpression:
            return data_map_func([xi])

        def map2(xi: ParameterExpression, xj: ParameterExpression) -> ParameterExpression:
            return data_map_func([xi, xj])

    # Phase‑shift re‑parameterisation
    if phase_shift_func is None:
        def shift(xi: ParameterExpression) -> ParameterExpression:
            return xi
    else:
        def shift(xi: ParameterExpression) -> ParameterExpression:
            return phase_shift_func(xi)

    # Shared pair scaling factor normalised by sqrt(n)
    pair_scale = base_pair_scale / sqrt(n)

    pairs = _resolve_entanglement(n, entanglement)

    for rep in range(int(reps)):
        qc.h(range(n))
        if insert_barriers:
            qc.barrier()
        for i in range(n):
            qc.p(2 * map1(shift(x[i])), i)
        if insert_barriers:
            qc.barrier()
        for (i, j) in pairs:
            qc.rzz(2 * pair_scale * map2(shift(x[i]), shift(x[j])), i, j)
        if insert_barriers and rep!= reps - 1:
            qc.barrier()

    qc.input_params = x  # type: ignore[attr-defined]
    return qc


class ZZFeatureMapRZZControlled(QuantumCircuit):
    """Class‑style wrapper for the controlled‑modified RZZ feature map."""

    def __init__(
        self,
        feature_dimension: int,
        reps: int = 2,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
        data_map_func: Callable[[Sequence[ParameterExpression]], ParameterExpression] | None = None,
        parameter_prefix: str = "x",
        insert_barriers: bool = False,
        base_pair_scale: float = 1.0,
        phase_shift_func: Callable[[ParameterExpression], ParameterExpression] | None = None,
        name: str = "ZZFeatureMapRZZControlled",
    ) -> None:
        built = zz_feature_map_rzz_controlled(
            feature_dimension,
            reps,
            entanglement,
            data_map_func,
            parameter_prefix,
            insert_barriers,
            base_pair_scale,
            phase_shift_func,
            name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)
        self.input_params = built.input_params  # type: ignore[attr-defined]


__all__ = ["ZZFeatureMapRZZControlled", "zz_feature_map_rzz_controlled"]
