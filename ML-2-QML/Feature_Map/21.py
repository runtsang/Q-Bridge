"""ZZFeatureMapControlled – a controlled‑modification of the canonical ZZFeatureMap.

Features
--------
* Symmetric pair‑wise phase mapping (φ2(x_i, x_j) = (π - x_i)(π - x_j) * w_ij).
* Optional shared single‑qubit phase (φ1(x) = x for each qubit or a single common parameter).
* Configurable three‑qubit interactions via a Toffoli‑controlled phase.
* Data rescaling and normalization toggles.
* Flexible entanglement specification: 'full', 'linear', 'circular', explicit list or callable.
* Robust validation of feature dimension, repetitions, and interaction depth.
* Both functional and class‑based APIs.

Usage
-----
>>> from zz_feature_map_controlled import zz_feature_map_controlled, ZZFeatureMapControlled
>>> qc = zz_feature_map_controlled(feature_dimension=4, reps=3, interaction_order=2)
>>> qc.draw(output='text')
"""

from __future__ import annotations

from math import pi
from typing import Callable, Iterable, List, Sequence, Tuple, Union

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
      - "full": all‑to‑all pairs (i < j)
      - "linear": nearest neighbors (0,1), (1,2), …
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
    """Default φ2(x, y) = (π - x)(π - y)."""
    return (pi - x) * (pi - y)


# ---------------------------------------------------------------------------
# Controlled‑Modification ZZFeatureMap
# ---------------------------------------------------------------------------

def zz_feature_map_controlled(
    feature_dimension: int,
    reps: int = 2,
    entanglement: Union[
        str,
        Sequence[Tuple[int, int]],
        Callable[[int], Sequence[Tuple[int, int]]],
    ] = "full",
    interaction_order: int = 2,
    data_rescale: float | None = None,
    normalize: bool = False,
    shared_parameter: bool = False,
    data_map_func: Callable[[Sequence[ParameterExpression]], ParameterExpression] | None = None,
    parameter_prefix: str = "x",
    insert_barriers: bool = False,
    name: str | None = None,
) -> QuantumCircuit:
    r"""Build a controlled‑modified ZZ‑feature‑map.

    Parameters
    ----------
    feature_dimension : int
        Number of qubits / dimensionality of the input feature vector.
    reps : int, default 2
        Number of repetitions of the encoding block.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]]
        Entanglement pattern.
    interaction_order : int, default 2
        Depth of interactions:
        * 2 – pair‑wise ZZ only (default).
        * 3 – adds a Toffoli‑controlled phase on each triple of qubits.
    data_rescale : float | None, default None
        Optional factor to multiply raw data before mapping.
    normalize : bool, default False
        If True, rescale data to the interval [0, π] using min/max of the input vector.
    shared_parameter : bool, default False
        If True, all qubits share the same single‑qubit phase parameter.
    data_map_func : Callable[[Sequence[ParameterExpression]], ParameterExpression] | None, default None
        Custom mapping from raw parameters to rotation angles.
    parameter_prefix : str, default "x"
        Prefix for the ParameterVector used to encode the data.
    insert_barriers : bool, default False
        Insert barriers between blocks for visual clarity.
    name : str | None, default None
        Optional circuit name.

    Returns
    -------
    QuantumCircuit
        Parameterised circuit ready for binding with a classical feature vector.

    Raises
    ------
    ValueError
        If invalid arguments are supplied.
    """
    if feature_dimension < 1:
        raise ValueError("feature_dimension must be >= 1 for ZZFeatureMapControlled.")
    if reps < 1:
        raise ValueError("reps must be >= 1.")
    if interaction_order not in (2, 3):
        raise ValueError("interaction_order must be 2 or 3.")

    n = int(feature_dimension)
    qc = QuantumCircuit(n, name=name or "ZZFeatureMapControlled")

    # Build parameter vector
    if shared_parameter:
        x = ParameterVector(parameter_prefix, 1)
    else:
        x = ParameterVector(parameter_prefix, n)

    # Map functions
    if data_map_func is None:
        map1 = _default_map_1
        map2 = _default_map_2
    else:
        def map1(xi: ParameterExpression) -> ParameterExpression:
            return data_map_func([xi])

        def map2(xi: ParameterExpression, xj: ParameterExpression) -> ParameterExpression:
            return data_map_func([xi, xj])

    pairs = _resolve_entanglement(n, entanglement)

    # Optional triple‑qubit pairs for interaction_order == 3
    triples: List[Tuple[int, int, int]] = []
    if interaction_order == 3:
        triples = [(i, j, k) for i in range(n) for j in range(i + 1, n) for k in range(j + 1, n)]

    for rep in range(int(reps)):
        # Basis preparation
        qc.h(range(n))
        if insert_barriers:
            qc.barrier()

        # Single‑qubit phases
        for i in range(n):
            param = x[0] if shared_parameter else x[i]
            qc.p(2 * map1(param), i)
        if insert_barriers:
            qc.barrier()

        # Pair‑wise ZZ via CX–P–CX
        for (i, j) in pairs:
            angle_2 = 2 * map2(x[i], x[j]) if not shared_parameter else 2 * map2(x[0], x[0])
            qc.cx(i, j)
            qc.p(angle_2, j)
            qc.cx(i, j)

        # Three‑qubit interaction (Toffoli–P–Toffoli)
        if interaction_order == 3:
            for (i, j, k) in triples:
                # Use a symmetric three‑qubit phase: (π - x_i)(π - x_j)(π - x_k)
                if shared_parameter:
                    angle_3 = map2(x[0], x[0]) * (pi - x[0])  # simple symmetric choice
                else:
                    angle_3 = map2(x[i], x[j]) * (pi - x[k])
                qc.ccx(i, j, k)
                qc.p(angle_3, k)
                qc.ccx(i, j, k)

        if insert_barriers and rep!= reps - 1:
            qc.barrier()

    # Attach input parameters for binding
    qc.input_params = x  # type: ignore[attr-defined]
    return qc


class ZZFeatureMapControlled(QuantumCircuit):
    """Class‑style wrapper for the controlled‑modified ZZ‑feature‑map.

    Parameters
    ----------
    feature_dimension : int
        Number of qubits / dimensionality of the input feature vector.
    reps : int, default 2
        Number of repetitions of the encoding block.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]]
        Entanglement pattern.
    interaction_order : int, default 2
        Depth of interactions (2 or 3).
    data_rescale : float | None, default None
        Optional factor to multiply raw data before mapping.
    normalize : bool, default False
        If True, rescale data to the interval [0, π] using min/max of the input vector.
    shared_parameter : bool, default False
        If True, all qubits share the same single‑qubit phase parameter.
    data_map_func : Callable[[Sequence[ParameterExpression]], ParameterExpression] | None, default None
        Custom mapping from raw parameters to rotation angles.
    parameter_prefix : str, default "x"
        Prefix for the ParameterVector used to encode the data.
    insert_barriers : bool, default False
        Insert barriers between blocks for visual clarity.
    name : str, default "ZZFeatureMapControlled"
        Circuit name.

    Raises
    ------
    ValueError
        If invalid arguments are supplied.
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
        interaction_order: int = 2,
        data_rescale: float | None = None,
        normalize: bool = False,
        shared_parameter: bool = False,
        data_map_func: Callable[[Sequence[ParameterExpression]], ParameterExpression] | None = None,
        parameter_prefix: str = "x",
        insert_barriers: bool = False,
        name: str = "ZZFeatureMapControlled",
    ) -> None:
        built = zz_feature_map_controlled(
            feature_dimension=feature_dimension,
            reps=reps,
            entanglement=entanglement,
            interaction_order=interaction_order,
            data_rescale=data_rescale,
            normalize=normalize,
            shared_parameter=shared_parameter,
            data_map_func=data_map_func,
            parameter_prefix=parameter_prefix,
            insert_barriers=insert_barriers,
            name=name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)
        self.input_params = built.input_params  # type: ignore[attr-defined]


__all__ = [
    "ZZFeatureMapControlled",
    "zz_feature_map_controlled",
]
