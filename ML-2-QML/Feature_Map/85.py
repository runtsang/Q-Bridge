"""Symmetrised, parameter‑shared ZZ feature map with optional data re‑scaling."""
from __future__ import annotations

from math import pi
from typing import Callable, Iterable, List, Sequence, Tuple

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterExpression, ParameterVector


# --------------------------------------------------------------------------- #
# Entanglement resolution
# --------------------------------------------------------------------------- #
def _resolve_entanglement(
    num_qubits: int,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Iterable[Tuple[int, int]]],
) -> List[Tuple[int, int]]:
    """
    Resolve an entanglement specification into a list of distinct qubit pairs.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    entanglement : str | Sequence[Tuple[int, int]] | Callable
        * ``"full"`` – all‑to‑all pairs (i < j)
        * ``"linear"`` – nearest‑neighbour chain
        * ``"circular"`` – linear plus wrap‑around between last and first qubit
        * explicit list of (i, j) tuples
        * callable f(num_qubits) → iterable of (i, j)

    Returns
    -------
    List[Tuple[int, int]]
        Validated list of qubit pairs.

    Raises
    ------
    ValueError
        If an invalid specification or out‑of‑range indices are provided.
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
        return [(int(i), int(j)) for i, j in pairs]

    # sequence of (i, j) pairs
    pairs = [(int(i), int(j)) for i, j in entanglement]  # type: ignore[arg-type]
    for i, j in pairs:
        if i == j:
            raise ValueError("Entanglement pairs must connect distinct qubits.")
        if not (0 <= i < num_qubits and 0 <= j < num_qubits):
            raise ValueError(f"Entanglement pair {(i, j)} out of range for n={num_qubits}.")
    return pairs


# --------------------------------------------------------------------------- #
# Default mapping functions
# --------------------------------------------------------------------------- #
def _default_single_map(x: ParameterExpression) -> ParameterExpression:
    """Default single‑qubit data map: φ₁(x) = x."""
    return x


def _default_pair_map(x: ParameterExpression, y: ParameterExpression) -> ParameterExpression:
    """Default two‑qubit data map: φ₂(x, y) = (π − x)(π − y)."""
    return (pi - x) * (pi - y)


# --------------------------------------------------------------------------- #
# Helper function
# --------------------------------------------------------------------------- #
def zz_feature_map_controlled(
    feature_dimension: int,
    reps: int = 2,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Iterable[Tuple[int, int]]] = "full",
    data_map_func: Callable[[Sequence[ParameterExpression]], ParameterExpression] | None = None,
    data_scaling: Callable[[Sequence[float]], Sequence[float]] | float | None = None,
    pre_rotation_func: Callable[[ParameterExpression], ParameterExpression] | None = None,
    parameter_prefix: str = "x",
    pair_prefix: str = "z",
    insert_barriers: bool = False,
    name: str | None = None,
) -> QuantumCircuit:
    """
    Build a symmetrised ZZ‑feature‑map with shared interaction parameters.

    The circuit follows the canonical Hadamard‑pre‑phase → CZ‑ZZ entanglement pattern,
    but each two‑qubit ZZ coupling shares a single parameter across all repetitions.
    Optional data‑re‑scaling and a user‑defined pre‑rotation are supported.

    Parameters
    ----------
    feature_dimension : int
        Number of classical features / qubits (must be ≥ 2).
    reps : int, default 2
        Number of repetitions of the feature‑map block.
    entanglement : str | Sequence[Tuple[int, int]] | Callable
        Entanglement specification (see :func:`_resolve_entanglement`).
    data_map_func : Callable[[Sequence[ParameterExpression]], ParameterExpression], optional
        Function mapping a list of ``ParameterExpression`` objects to a new expression.
        If ``None``, defaults to :func:`_default_single_map` for single‑qubit
        phases and :func:`_default_pair_map` for pair interactions.
    data_scaling : Callable[[Sequence[float]], Sequence[float]] | float, optional
        Optional scaling applied to classical data before binding.
        If a float, it is used as a multiplicative factor.
        If a callable, it is applied element‑wise to the data vector.
        The scaling is stored on the circuit for reference but does not alter the
        symbolic parameters themselves.
    pre_rotation_func : Callable[[ParameterExpression], ParameterExpression], optional
        Function providing a pre‑rotation angle for each qubit.
        If ``None``, the identity (no extra rotation) is used.
    parameter_prefix : str, default "x"
        Prefix for the data parameters.
    pair_prefix : str, default "z"
        Prefix for the shared interaction parameters.
    insert_barriers : bool, default False
        Whether to insert barriers between layers for clearer visualisation.
    name : str | None, default None
        Optional circuit name; defaults to ``"ZZFeatureMapControlled"``.

    Returns
    -------
    QuantumCircuit
        The constructed feature‑map circuit.

    Raises
    ------
    ValueError
        If input arguments are invalid.
    """
    if feature_dimension < 2:
        raise ValueError("feature_dimension must be >= 2.")
    if reps < 1:
        raise ValueError("reps must be a positive integer.")

    n = int(feature_dimension)
    qc = QuantumCircuit(n, name=name or "ZZFeatureMapControlled")

    # Data parameters
    x = ParameterVector(parameter_prefix, n)

    # Interaction parameters – one per qubit pair
    pairs = _resolve_entanglement(n, entanglement)
    pair_params = ParameterVector(pair_prefix, len(pairs))

    # Map functions
    if data_map_func is None:
        single_map = _default_single_map
        pair_map = _default_pair_map
    else:
        # Wrap to accept the same signature as the defaults
        def single_map(xi: ParameterExpression) -> ParameterExpression:
            return data_map_func([xi])

        def pair_map(xi: ParameterExpression, xj: ParameterExpression) -> ParameterExpression:
            return data_map_func([xi, xj])

    # Pre‑rotation
    if pre_rotation_func is None:
        pre_rot = lambda xi: xi
    else:
        pre_rot = pre_rotation_func

    for rep in range(int(reps)):
        # Basis preparation
        qc.h(range(n))
        if insert_barriers:
            qc.barrier()

        # Single‑qubit pre‑rotations
        for i in range(n):
            qc.p(2 * pre_rot(x[i]), i)

        if insert_barriers:
            qc.barrier()

        # Shared‑parameter ZZ entanglers
        for idx, (i, j) in enumerate(pairs):
            angle = 2 * pair_params[idx] * pair_map(x[i], x[j])
            qc.cx(i, j)
            qc.p(angle, j)
            qc.cx(i, j)

        if insert_barriers and rep!= reps - 1:
            qc.barrier()

    # Attach parameter vectors for convenience
    qc.input_params = x  # type: ignore[attr-defined]
    qc.interaction_params = pair_params  # type: ignore[attr-defined]

    # Store optional scaling for reference
    qc.data_scaling = data_scaling  # type: ignore[attr-defined]

    return qc


# --------------------------------------------------------------------------- #
# Object‑oriented wrapper
# --------------------------------------------------------------------------- #
class ZZFeatureMapControlled(QuantumCircuit):
    """
    OO wrapper for :func:`zz_feature_map_controlled`.

    Instantiation parameters match the helper function.  The wrapped circuit
    exposes ``input_params`` and ``interaction_params`` attributes for
    direct access to the symbolic parameters.

    Examples
    --------
    >>> from qiskit import transpile, Aer
    >>> from qiskit.circuit import ParameterVector
    >>> from qiskit.utils import QuantumInstance
    >>> from qiskit.providers.aer import AerSimulator
    >>> import numpy as np
    >>> # Build the circuit
    >>> fmap = ZZFeatureMapControlled(feature_dimension=4, reps=1)
    >>> # Bind parameters
    >>> data = np.random.rand(4)
    >>> bound = fmap.bind_parameters({p: d for p, d in zip(fmap.input_params, data)})
    >>> # Execute
    >>> backend = AerSimulator()
    >>> result = backend.run(bound).result()
    >>> counts = result.get_counts()
    """

    def __init__(
        self,
        feature_dimension: int,
        reps: int = 2,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Iterable[Tuple[int, int]]] = "full",
        data_map_func: Callable[[Sequence[ParameterExpression]], ParameterExpression] | None = None,
        data_scaling: Callable[[Sequence[float]], Sequence[float]] | float | None = None,
        pre_rotation_func: Callable[[ParameterExpression], ParameterExpression] | None = None,
        parameter_prefix: str = "x",
        pair_prefix: str = "z",
        insert_barriers: bool = False,
        name: str = "ZZFeatureMapControlled",
    ) -> None:
        built = zz_feature_map_controlled(
            feature_dimension=feature_dimension,
            reps=reps,
            entanglement=entanglement,
            data_map_func=data_map_func,
            data_scaling=data_scaling,
            pre_rotation_func=pre_rotation_func,
            parameter_prefix=parameter_prefix,
            pair_prefix=pair_prefix,
            insert_barriers=insert_barriers,
            name=name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)
        self.input_params = built.input_params  # type: ignore[attr-defined]
        self.interaction_params = built.interaction_params  # type: ignore[attr-defined]
        self.data_scaling = built.data_scaling  # type: ignore[attr-defined]


__all__ = [
    "ZZFeatureMapControlled",
    "zz_feature_map_controlled",
]
