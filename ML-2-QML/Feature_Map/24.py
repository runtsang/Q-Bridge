from __future__ import annotations

from math import pi
from typing import Callable, List, Sequence, Tuple, Union

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterExpression, ParameterVector


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _resolve_entanglement(
    num_qubits: int,
    entanglement: Union[
        str,
        Sequence[Tuple[int, int]],
        Callable[[int], Sequence[Tuple[int, int]]],
    ],
) -> List[Tuple[int, int]]:
    """Return a list of two‑qubit pairs according to a simple entanglement spec.

    Supported specs:
      - "full"      : all-to-all pairs (i < j)
      - "linear"    : nearest neighbours (0,1), (1,2), …
      - "circular"  : linear plus wrap‑around (n-1,0) if n > 2
      - "symmetrized": pair each qubit with its mirror across the centre
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
        if entanglement == "symmetrized":
            pairs = []
            for i in range(num_qubits // 2):
                j = num_qubits - 1 - i
                if i < j:
                    pairs.append((i, j))
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
    """Default single‑qubit phase map φ1(x) = x."""
    return x


def _default_map_2(x: ParameterExpression, y: ParameterExpression) -> ParameterExpression:
    """Default two‑qubit phase map φ2(x, y) = (π − x)(π − y)."""
    return (pi - x) * (pi - y)


# ---------------------------------------------------------------------------
# Canonical ZZFeatureMap (CX–P–CX for ZZ) – controlled modification
# ---------------------------------------------------------------------------

def zz_feature_map(
    feature_dimension: int,
    reps: int = 2,
    entanglement: Union[
        str,
        Sequence[Tuple[int, int]],
        Callable[[int], Sequence[Tuple[int, int]]],
    ] = "full",
    data_map_func: Callable[[Sequence[ParameterExpression]], ParameterExpression] | None = None,
    shared_params: bool = False,
    parameter_prefix: str = "x",
    insert_barriers: bool = False,
    name: str | None = None,
) -> QuantumCircuit:
    """Build a controlled‑modification ZZ‑feature‑map.

    Parameters
    ----------
    feature_dimension : int
        Number of qubits / data features. Must be >= 2.
    reps : int, default 2
        Number of circuit repetitions (layers). Each layer contains
        H → P(2·φ1) → ZZ entanglers via CX–P–CX.
    entanglement : str | Sequence[Tuple[int, int]] | Callable
        Entanglement topology. See :func:`_resolve_entanglement`.
    data_map_func : Callable[[Sequence[ParameterExpression]], ParameterExpression] | None
        Optional user‑supplied mapping from raw feature values to phase angles.
        If ``None`` the default behaviour is used.
    shared_params : bool, default False
        When ``True`` a single parameter is shared across all qubits and
        across all repetitions, providing a highly symmetric encoding.
    parameter_prefix : str, default "x"
        Prefix used for automatically generated parameter names.
    insert_barriers : bool, default False
        Whether to insert barriers between logical blocks for improved
        circuit readability and to aid transpiler optimisation.
    name : str | None, default None
        Name of the resulting :class:`QuantumCircuit`. If ``None`` a default
        name is chosen.

    Returns
    -------
    QuantumCircuit
        Fully built feature‑map circuit.

    Raises
    ------
    ValueError
        If ``feature_dimension`` is less than 2, or if an invalid
        entanglement specification is supplied.
    """
    if feature_dimension < 2:
        raise ValueError("feature_dimension must be >= 2 for ZZFeatureMap.")

    n = int(feature_dimension)
    qc = QuantumCircuit(n, name=name or "ZZFeatureMap")

    # Parameter handling
    if shared_params:
        param_vec = ParameterVector(parameter_prefix, 1)
        x_shared = param_vec[0]
        x = [x_shared for _ in range(n)]
    else:
        param_vec = ParameterVector(parameter_prefix, n)
        x = param_vec

    # Mapping functions
    if data_map_func is None:
        map1 = _default_map_1
        map2 = _default_map_2
    else:
        def map1(xi: ParameterExpression) -> ParameterExpression:
            return data_map_func([xi])

        def map2(xi: ParameterExpression, xj: ParameterExpression) -> ParameterExpression:
            return data_map_func([xi, xj])

    # Entanglement pairs
    pairs = _resolve_entanglement(n, entanglement)

    # Build layers
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

        # ZZ via CX–P–CX
        for (i, j) in pairs:
            angle_2 = 2 * map2(x[i], x[j])
            qc.cx(i, j)
            qc.p(angle_2, j)
            qc.cx(i, j)

        if insert_barriers and rep!= reps - 1:
            qc.barrier()

    qc.input_params = param_vec  # type: ignore[attr-defined]
    return qc


class ZZFeatureMap(QuantumCircuit):
    """Class‑style wrapper for the controlled‑modification ZZ‑feature‑map.

    The constructor simply forwards all arguments to :func:`zz_feature_map`
    and then composes the resulting circuit onto the subclass instance.
    """
    def __init__(
        self,
        feature_dimension: int,
        reps: int = 2,
        entanglement: Union[
            str,
            Sequence[Tuple[int, int]],
            Callable[[int], Sequence[Tuple[int, int]]],
        ] = "full",
        data_map_func: Callable[[Sequence[ParameterExpression]], ParameterExpression] | None = None,
        shared_params: bool = False,
        parameter_prefix: str = "x",
        insert_barriers: bool = False,
        name: str = "ZZFeatureMap",
    ) -> None:
        built = zz_feature_map(
            feature_dimension=feature_dimension,
            reps=reps,
            entanglement=entanglement,
            data_map_func=data_map_func,
            shared_params=shared_params,
            parameter_prefix=parameter_prefix,
            insert_barriers=insert_barriers,
            name=name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)
        self.input_params = built.input_params  # type: ignore[attr-defined]


__all__ = ["ZZFeatureMap", "zz_feature_map"]
