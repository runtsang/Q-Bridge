from __future__ import annotations

from math import pi
from typing import Callable, List, Sequence, Tuple

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
      - "linear": nearest‑neighbor pairs
      - "circular": linear plus wrap‑around
      - explicit list of pairs
      - callable returning a sequence of pairs
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
    pairs = [(int(i), int(j)) for (i, j) in entanglement]
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

# ---------------------------------------------------------------------------
# Controlled‑modification feature map
# ---------------------------------------------------------------------------

def zz_feature_map_rzz_controlled(
    feature_dimension: int,
    reps: int = 2,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
    data_map_func: Callable[[Sequence[ParameterExpression]], ParameterExpression] | None = None,
    parameter_prefix: str = "x",
    insert_barriers: bool = False,
    pair_scale: float = 1.0,
    shared_pair_params: bool = False,
    name: str | None = None,
) -> QuantumCircuit:
    """
    Controlled‑modification of the RZZ‑entangled feature map.

    The circuit retains the same layer structure as the original seed but introduces
    a *shared* pair parameter when ``shared_pair_params=True``.  When
    ``shared_pair_params=False`` each pair receives its own parameter, increasing
    the expressive power.

    Parameters
    ----------
    feature_dimension : int
        Number of classical features and qubits.  Must be >= 2.
    reps : int, default=2
        Number of repetitions of the H‑P‑RZZ block.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]]
        Specifies which qubit pairs are entangled.
    data_map_func : Callable[[Sequence[ParameterExpression]], ParameterExpression] | None, default=None
        Maps raw feature values to rotation angles.  If ``None`` the defaults
        φ1(x)=x and φ2(x,y)=(π−x)(π−y) are used.
    parameter_prefix : str, default="x"
        Prefix for the parameter vector representing the input features.
    insert_barriers : bool, default=False
        If True, inserts barriers after each layer for easier visualisation.
    pair_scale : float, default=1.0
        Global scaling factor applied to all RZZ angles.
    shared_pair_params : bool, default=False
        When ``True`` a single pair parameter is shared across all qubit pairs.
        When ``False`` each pair receives an independent parameter.
    name : str | None, default=None
        Optional circuit name.

    Returns
    -------
    QuantumCircuit
        The constructed feature‑map circuit.
    """
    if feature_dimension < 2:
        raise ValueError("feature_dimension must be >= 2.")
    n = int(feature_dimension)
    qc = QuantumCircuit(n, name=name or "ZZFeatureMapRZZControlled")

    # Parameter vector for the input features
    x = ParameterVector(parameter_prefix, n)

    # Define mapping functions
    if data_map_func is None:
        map1 = _default_map_1
        map2 = _default_map_2
    else:
        def map1(xi: ParameterExpression) -> ParameterExpression:
            return data_map_func([xi])

        def map2(xi: ParameterExpression, xj: ParameterExpression) -> ParameterExpression:
            return data_map_func([xi, xj])

    pairs = _resolve_entanglement(n, entanglement)
    num_pairs = len(pairs)

    # Parameter vector for pair couplings
    pair_params = ParameterVector("p", 1 if shared_pair_params else num_pairs)

    for rep in range(int(reps)):
        # Single‑qubit layer
        qc.h(range(n))
        if insert_barriers:
            qc.barrier()
        for i in range(n):
            qc.p(2 * map1(x[i]), i)
        if insert_barriers:
            qc.barrier()

        # Two‑qubit entanglement
        for k, (i, j) in enumerate(pairs):
            pair_angle = pair_params[0] if shared_pair_params else pair_params[k]
            qc.rzz(2 * pair_scale * pair_angle * map2(x[i], x[j]), i, j)
        if insert_barriers and rep!= reps - 1:
            qc.barrier()

    qc.input_params = x  # type: ignore[attr-defined]
    return qc


class ZZFeatureMapRZZControlled(QuantumCircuit):
    """
    OO wrapper for the controlled‑modification feature map.

    Parameters are identical to :func:`zz_feature_map_rzz_controlled`.  The
    constructed circuit is composed into the instance.
    """
    def __init__(
        self,
        feature_dimension: int,
        reps: int = 2,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
        data_map_func: Callable[[Sequence[ParameterExpression]], ParameterExpression] | None = None,
        parameter_prefix: str = "x",
        insert_barriers: bool = False,
        pair_scale: float = 1.0,
        shared_pair_params: bool = False,
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
            shared_pair_params,
            name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)
        self.input_params = built.input_params  # type: ignore[attr-defined]


__all__ = ["ZZFeatureMapRZZControlled", "zz_feature_map_rzz_controlled"]
