"""ControlledZZFeatureMap builder with shared parameters and optional rotations."""
from __future__ import annotations

from math import pi
from typing import Callable, Sequence, Tuple, List, Optional

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
      - ``"full"``: all‑to‑all pairs (i < j)
      - ``"linear"``: nearest neighbours (0,1), (1,2), …
      - ``"circular"``: linear plus wrap‑around (n-1,0) if n > 2
      - explicit list of pairs like [(0, 2), (1, 3)]
      - callable: f(n) -> sequence of (i, j)
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
        return [(int(i), int(j)) for i, j in entanglement(num_qubits)]

    return [(int(i), int(j)) for i, j in entanglement]  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Controlled ZZFeatureMap
# ---------------------------------------------------------------------------

def controlled_zz_feature_map(
    feature_dimension: int,
    reps: int = 2,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "circular",
    data_map_func: Callable[[Sequence[ParameterExpression]], ParameterExpression] | None = None,
    shared_zz: bool = False,
    shared_phase: bool = False,
    pre_rotation: bool = False,
    post_rotation: bool = False,
    normalise: bool = False,
    parameter_prefix: str = "x",
    insert_barriers: bool = False,
    name: str | None = None,
) -> QuantumCircuit:
    """
    Build a controlled‑parameterised ZZ‑feature‑map.

    The circuit follows the canonical CX–P–CX pattern for ZZ interactions but adds
    several controlled modifications:

    - **Shared parameters**: ``shared_phase`` and ``shared_zz`` enable a single
      global parameter for all single‑qubit phases or all ZZ angles, respectively.
    - **Optional rotations**: ``pre_rotation`` and ``post_rotation`` add a global
      Y‑rotation (π/4) before/after each repetition.
    - **Entanglement flexibility**: ``entanglement`` may be ``full``, ``linear``,
      ``circular``, a custom list of pairs, or a callable.
    - **Data re‑parameterisation**: ``data_map_func`` can replace the default
      mapping functions.
    - **Normalisation**: If ``normalise`` is ``True``, the user is expected to
      provide a normalised feature vector; the circuit itself does not alter the
      data.

    Parameters
    ----------
    feature_dimension : int
        Number of input features (qubits). Must be >= 2.
    reps : int, default 2
        Number of repetitions of the encoding block.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]]
        Specification of qubit pairs to entangle.
    data_map_func : Callable[[Sequence[ParameterExpression]], ParameterExpression] | None
        Custom mapping from feature vector to a parameter expression.
    shared_zz : bool, default False
        Use a single parameter for all ZZ angles.
    shared_phase : bool, default False
        Use a single parameter for all single‑qubit phases.
    pre_rotation : bool, default False
        Add a global Y‑rotation (π/4) before each repetition.
    post_rotation : bool, default False
        Add a global Y‑rotation (π/4) after each repetition.
    normalise : bool, default False
        If ``True``, the caller must supply a normalised feature vector.
    parameter_prefix : str, default "x"
        Prefix for the feature parameters.
    insert_barriers : bool, default False
        Insert barriers between blocks for readability.
    name : str | None, default None
        Circuit name.

    Returns
    -------
    QuantumCircuit
        Parameterised circuit ready for binding with a classical feature vector.

    Raises
    ------
    ValueError
        If ``feature_dimension`` < 2 or ``reps`` <= 0.
    """
    if feature_dimension < 2:
        raise ValueError("feature_dimension must be >= 2 for ControlledZZFeatureMap.")
    if reps <= 0:
        raise ValueError("reps must be a positive integer.")

    n = int(feature_dimension)
    qc = QuantumCircuit(n, name=name or "ControlledZZFeatureMap")

    # Feature parameters
    x = ParameterVector(parameter_prefix, n)

    # Optional shared parameters
    shared_params: List[ParameterVector] = []
    if shared_phase:
        alpha = ParameterVector("alpha", 1)
        shared_params.append(alpha)
    if shared_zz:
        beta = ParameterVector("beta", 1)
        shared_params.append(beta)

    # Mapping functions
    if data_map_func is None:
        def map1(xi: ParameterExpression) -> ParameterExpression:
            return alpha[0] * xi if shared_phase else xi

        def map2(xi: ParameterExpression, xj: ParameterExpression) -> ParameterExpression:
            base = (pi - xi) * (pi - xj)
            return beta[0] * base if shared_zz else base
    else:
        def map1(xi: ParameterExpression) -> ParameterExpression:
            return data_map_func([xi])

        def map2(xi: ParameterExpression, xj: ParameterExpression) -> ParameterExpression:
            return data_map_func([xi, xj])

    pairs = _resolve_entanglement(n, entanglement)

    for rep in range(reps):
        if pre_rotation:
            qc.ry(pi / 4, range(n))
        qc.h(range(n))
        if insert_barriers:
            qc.barrier()
        for i in range(n):
            qc.p(2 * map1(x[i]), i)
        if insert_barriers:
            qc.barrier()
        for (i, j) in pairs:
            angle_2 = 2 * map2(x[i], x[j])
            qc.cx(i, j)
            qc.p(angle_2, j)
            qc.cx(i, j)
        if post_rotation:
            qc.ry(pi / 4, range(n))
        if insert_barriers and rep!= reps - 1:
            qc.barrier()

    # Store all parameters for binding
    qc.input_params = [x] + shared_params  # type: ignore[attr-defined]
    return qc


class ControlledZZFeatureMap(QuantumCircuit):
    """Class‑style wrapper for ControlledZZFeatureMap."""

    def __init__(
        self,
        feature_dimension: int,
        reps: int = 2,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "circular",
        data_map_func: Callable[[Sequence[ParameterExpression]], ParameterExpression] | None = None,
        shared_zz: bool = False,
        shared_phase: bool = False,
        pre_rotation: bool = False,
        post_rotation: bool = False,
        normalise: bool = False,
        parameter_prefix: str = "x",
        insert_barriers: bool = False,
        name: str = "ControlledZZFeatureMap",
    ) -> None:
        built = controlled_zz_feature_map(
            feature_dimension=feature_dimension,
            reps=reps,
            entanglement=entanglement,
            data_map_func=data_map_func,
            shared_zz=shared_zz,
            shared_phase=shared_phase,
            pre_rotation=pre_rotation,
            post_rotation=post_rotation,
            normalise=normalise,
            parameter_prefix=parameter_prefix,
            insert_barriers=insert_barriers,
            name=name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)
        self.input_params = built.input_params  # type: ignore[attr-defined]


__all__ = ["ControlledZZFeatureMap", "controlled_zz_feature_map"]
