"""Feature map with controlled modifications:
- Shared ZZ interaction parameter to reduce parameter count.
- Symmetrised entanglement pattern (double CX–P–CX per pair).
- Optional normalisation of input features to [0, π].
- Maintains compatibility with Qiskit data‑encoding workflows.

The module exposes both a functional helper `zz_feature_map` and an OO wrapper `ZZFeatureMap`.
"""
from __future__ import annotations

from math import pi
from typing import Callable, Iterable, Sequence, Tuple

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterExpression, ParameterVector


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _resolve_entanglement(
    num_qubits: int,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]],
) -> list[Tuple[int, int]]:
    """Return a list of two‑qubit pairs according to the supplied spec.

    Supported specs:

    * ``"full"``     – all‑to‑all pairs (i < j)
    * ``"linear"``   – nearest‑neighbour chain
    * ``"circular"`` – linear + wrap‑around (n‑1,0) if n > 2
    * ``"symmetrised"`` – same as ``"full"`` but each pair is added twice
      (once as (i,j) and once as (j,i)) to double the interaction depth.
    * explicit list of pairs
    * callable ``f(num_qubits) -> sequence``

    Raises
    ------
    ValueError
        If an unknown spec is supplied or a pair references an out‑of-range qubit.
    """
    if isinstance(entanglement, str):
        if entanglement == "full":
            pairs = [(i, j) for i in range(num_qubits) for j in range(i + 1, num_qubits)]
        elif entanglement == "linear":
            pairs = [(i, i + 1) for i in range(num_qubits - 1)]
        elif entanglement == "circular":
            pairs = [(i, i + 1) for i in range(num_qubits - 1)]
            if num_qubits > 2:
                pairs.append((num_qubits - 1, 0))
        elif entanglement == "symmetrised":
            base = _resolve_entanglement(num_qubits, "full")
            pairs = base + [(j, i) for (i, j) in base]
        else:
            raise ValueError(f"Unknown entanglement spec: {entanglement!r}")
    elif callable(entanglement):
        pairs = list(entanglement(num_qubits))
        pairs = [(int(i), int(j)) for (i, j) in pairs]
    else:
        pairs = [(int(i), int(j)) for (i, j) in entanglement]  # type: ignore[arg-type]

    # Basic validation
    for i, j in pairs:
        if i == j:
            raise ValueError("Entanglement pairs must connect distinct qubits.")
        if not (0 <= i < num_qubits and 0 <= j < num_qubits):
            raise ValueError(f"Entanglement pair {(i, j)} out of range for n={num_qubits}.")
    return pairs


# ---------------------------------------------------------------------------
# Default data‑mapping functions
# ---------------------------------------------------------------------------

def _default_map_1(x: ParameterExpression, normalize: bool = False) -> ParameterExpression:
    """φ1(x) = x, optionally scaled to [0,π]."""
    return pi * x if normalize else x


def _default_map_2(
    x: ParameterExpression,
    y: ParameterExpression,
    normalize: bool = False,
) -> ParameterExpression:
    """φ2(x,y) = (x + y)/2, optionally scaled to [0,π]."""
    avg = (x + y) / 2
    return pi * avg if normalize else avg


# ---------------------------------------------------------------------------
# Canonical (but modified) ZZ‑FeatureMap
# ---------------------------------------------------------------------------

def zz_feature_map(
    feature_dimension: int,
    reps: int = 2,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
    data_map_func: Callable[[Sequence[ParameterExpression]], ParameterExpression] | None = None,
    shared_zz_param: bool = False,
    parameter_prefix: str = "x",
    shared_parameter_prefix: str = "theta",
    insert_barriers: bool = False,
    normalize: bool = False,
    name: str | None = None,
) -> QuantumCircuit:
    """Build a modified ZZ‑feature‑map.

    The circuit follows the classic pattern

        H → P(2·φ1) → ZZ via CX–P–CX

    but with the following controlled modifications:

    * A single shared parameter (``theta``) multiplies all ZZ angles when
      ``shared_zz_param=True`` – drastically reducing the number of free
      parameters.
    * ``"symmetrised"`` entanglement doubles the interaction depth by applying
      CX–P–CX in both directions for each pair.
    * Optional normalisation of input features to the interval [0,π].
    * Optional barriers for visual clarity.

    Parameters
    ----------
    feature_dimension : int
        Number of input features / qubits. Must be >= 2.
    reps : int, default 2
        Number of repetitions of the feature‑map block.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]]
        Entanglement specification. See :func:`_resolve_entanglement` for details.
    data_map_func : Callable[[Sequence[ParameterExpression]], ParameterExpression] | None
        Custom mapping from input parameters to rotation angles.
        If ``None``, defaults to ``φ1(x)=x`` and ``φ2(x,y)=(x+y)/2``.
    shared_zz_param : bool, default False
        When ``True`` a single parameter ``theta`` multiplies all ZZ angles.
    parameter_prefix : str, default "x"
        Prefix for the per‑qubit data parameters.
    shared_parameter_prefix : str, default "theta"
        Prefix for the shared ZZ parameter.
    insert_barriers : bool, default False
        Whether to insert barriers between logical blocks.
    normalize : bool, default False
        Scale input features into [0,π] before mapping.
    name : str | None
        Optional circuit name.

    Returns
    -------
    QuantumCircuit
        The constructed feature‑map circuit. It exposes two attributes:

        * ``input_params`` – the ParameterVector for data inputs.
        * ``shared_params`` – the ParameterVector for the shared ZZ parameter
          (if ``shared_zz_param=True``).

    Raises
    ------
    ValueError
        If ``feature_dimension`` < 2 or if an invalid entanglement spec is supplied.
    """
    if feature_dimension < 2:
        raise ValueError("feature_dimension must be >= 2 for ZZFeatureMap.")

    n = int(feature_dimension)
    qc = QuantumCircuit(n, name=name or "ZZFeatureMap")

    # Data parameters per qubit
    data_params = ParameterVector(parameter_prefix, n)

    # Optional shared ZZ parameter
    shared_params = None
    if shared_zz_param:
        shared_params = ParameterVector(shared_parameter_prefix, 1)

    # Resolve entanglement pairs
    pairs = _resolve_entanglement(n, entanglement)

    # Choose data‑mapping functions
    if data_map_func is None:
        # Default symmetrical mapping
        map1 = lambda xi: _default_map_1(xi, normalize=normalize)
        map2 = lambda xi, xj: _default_map_2(xi, xj, normalize=normalize)
    else:
        def map1(xi: ParameterExpression) -> ParameterExpression:
            return data_map_func([xi])

        def map2(xi: ParameterExpression, xj: ParameterExpression) -> ParameterExpression:
            return data_map_func([xi, xj])

    for rep in range(int(reps)):
        # Basis preparation
        qc.h(range(n))
        if insert_barriers:
            qc.barrier()

        # Single‑qubit phases
        for i in range(n):
            qc.p(2 * map1(data_params[i]), i)

        if insert_barriers:
            qc.barrier()

        # ZZ entanglers via CX–P–CX
        for (i, j) in pairs:
            angle = 2 * map2(data_params[i], data_params[j])
            if shared_params:
                angle = angle * shared_params[0]
            qc.cx(i, j)
            qc.p(angle, j)
            qc.cx(i, j)

        if insert_barriers and rep!= reps - 1:
            qc.barrier()

    qc.input_params = data_params  # type: ignore[attr-defined]
    if shared_params:
        qc.shared_params = shared_params  # type: ignore[attr-defined]
    return qc


class ZZFeatureMap(QuantumCircuit):
    """Object‑oriented wrapper for the modified ZZ‑feature‑map.

    Parameters
    ----------
    Same as :func:`zz_feature_map`.  All keyword arguments are forwarded
    directly to the functional constructor.

    The resulting circuit inherits all attributes from :class:`QuantumCircuit`
    and exposes ``input_params`` (data parameters) and optionally
    ``shared_params`` (the shared ZZ parameter) for convenient binding.
    """
    def __init__(
        self,
        feature_dimension: int,
        reps: int = 2,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
        data_map_func: Callable[[Sequence[ParameterExpression]], ParameterExpression] | None = None,
        shared_zz_param: bool = False,
        parameter_prefix: str = "x",
        shared_parameter_prefix: str = "theta",
        insert_barriers: bool = False,
        normalize: bool = False,
        name: str = "ZZFeatureMap",
    ) -> None:
        built = zz_feature_map(
            feature_dimension=feature_dimension,
            reps=reps,
            entanglement=entanglement,
            data_map_func=data_map_func,
            shared_zz_param=shared_zz_param,
            parameter_prefix=parameter_prefix,
            shared_parameter_prefix=shared_parameter_prefix,
            insert_barriers=insert_barriers,
            normalize=normalize,
            name=name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)
        self.input_params = built.input_params  # type: ignore[attr-defined]
        if hasattr(built, "shared_params"):
            self.shared_params = built.shared_params  # type: ignore[attr-defined]


__all__ = ["ZZFeatureMap", "zz_feature_map"]
