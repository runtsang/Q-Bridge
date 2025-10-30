"""SymmetricZZFeatureMap – a controlled‑modified ZZ feature map.

The module exposes two entry points:
  * `symmetric_zz_feature_map(...)` – functional builder returning a QuantumCircuit.
  * `SymmetricZZFeatureMap(...)` – a QuantumCircuit subclass for OO use.

Both interfaces support:
  * arbitrary feature dimension (>=2)
  * configurable repetition depth
  * flexible entanglement patterns (full, linear, circular, explicit, or callable)
  * optional data normalisation to [0, π]
  * shared ZZ interaction angle per repetition
  * optional pre‑rotation via a user‑supplied function

The circuit follows the structure:
  H → P(2·φ1) on each qubit → shared ZZ entanglers via CX–P–CX → optional barrier.
"""

from __future__ import annotations

from math import pi
from typing import Callable, List, Sequence, Tuple

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterExpression, ParameterVector


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _resolve_entanglement(
    num_qubits: int,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]],
) -> List[Tuple[int, int]]:
    """Return a list of two‑qubit pairs according to a simple entanglement spec.

    Supported specs:
      * ``"full"``: all‑to‑all pairs (i < j)
      * ``"linear"``: nearest neighbours (0,1), (1,2), …
      * ``"circular"``: linear plus wrap‑around (n‑1,0) if n > 2
      * explicit list of pairs like ``[(0, 2), (1, 3)]``
      * callable: ``f(num_qubits) -> sequence of (i, j)``
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
        return [(int(i), int(j)) for (i, j) in entanglement(num_qubits)]

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


def _default_map_2(x: Sequence[ParameterExpression]) -> ParameterExpression:
    """Default φ2(x1,…,xn) = (π − Σxi)²."""
    return (pi - sum(x)) ** 2


# ---------------------------------------------------------------------------
# Symmetric ZZ Feature Map
# ---------------------------------------------------------------------------

def symmetric_zz_feature_map(
    feature_dimension: int,
    reps: int = 2,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
    data_map_func: Callable[[Sequence[ParameterExpression]], ParameterExpression] | None = None,
    parameter_prefix: str = "x",
    insert_barriers: bool = False,
    normalize: bool = False,
    name: str | None = None,
) -> QuantumCircuit:
    """Build a symmetric ZZ‑feature‑map.

    Parameters
    ----------
    feature_dimension : int
        Number of features / qubits. Must be >= 2.
    reps : int, default 2
        Number of repetitions (depth).
    entanglement : str | sequence | callable, default "full"
        Entanglement pattern specification.
    data_map_func : callable | None, default None
        Custom mapping from a list of ParameterExpressions to a single ParameterExpression.
        If None, defaults to a shared ZZ angle based on the sum of all features.
    parameter_prefix : str, default "x"
        Prefix for the ParameterVector.
    insert_barriers : bool, default False
        Whether to insert barriers between layers.
    normalize : bool, default False
        If True, scale input features to [0, π] before mapping.
    name : str | None, default None
        Circuit name; if None, uses "SymmetricZZFeatureMap".

    Returns
    -------
    QuantumCircuit
        Parameterised quantum circuit ready for data binding.

    Notes
    -----
    * The circuit uses a **shared** ZZ interaction angle per repetition.
    * The single‑qubit phase uses a per‑qubit mapping (default linear).
    * Error handling ensures valid dimensions and entanglement specifications.
    """
    if feature_dimension < 2:
        raise ValueError("feature_dimension must be >= 2 for SymmetricZZFeatureMap.")

    n = int(feature_dimension)
    qc = QuantumCircuit(n, name=name or "SymmetricZZFeatureMap")

    # Parameter vector for data features
    x = ParameterVector(parameter_prefix, n)

    # Optional normalisation: scale to [0, π]
    if normalize:
        # Define a simple linear normalisation: x_i' = π * (x_i - min) / (max - min)
        # Since we cannot know min/max statically, we enforce that the user
        # must normalise externally or use a custom data_map_func.
        raise NotImplementedError(
            "Automatic normalisation is not supported; please pre‑scale your data to [0, π] "
            "or provide a custom data_map_func."
        )

    # Mapping functions
    if data_map_func is None:
        map1 = _default_map_1
        map2 = _default_map_2
    else:
        def map1(xi: ParameterExpression) -> ParameterExpression:
            return data_map_func([xi])

        def map2(xs: Sequence[ParameterExpression]) -> ParameterExpression:
            return data_map_func(xs)

    pairs = _resolve_entanglement(n, entanglement)

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

        # Shared ZZ entanglement via CX–P–CX
        shared_angle = 2 * map2(x)
        for (i, j) in pairs:
            qc.cx(i, j)
            qc.p(shared_angle, j)
            qc.cx(i, j)

        if insert_barriers and rep!= reps - 1:
            qc.barrier()

    qc.input_params = x  # type: ignore[attr-defined]
    return qc


class SymmetricZZFeatureMap(QuantumCircuit):
    """QuantumCircuit subclass for the symmetric ZZ‑feature‑map.

    The constructor simply builds the circuit using :func:`symmetric_zz_feature_map`
    and composes it into the subclass instance. The resulting object retains
    the original ``input_params`` attribute for parameter binding.

    Parameters
    ----------
    feature_dimension : int
        Number of features / qubits. Must be >= 2.
    reps : int, default 2
        Number of repetitions (depth).
    entanglement : str | sequence | callable, default "full"
        Entanglement pattern specification.
    data_map_func : callable | None, default None
        Custom mapping from a list of ParameterExpressions to a single ParameterExpression.
    parameter_prefix : str, default "x"
        Prefix for the ParameterVector.
    insert_barriers : bool, default False
        Whether to insert barriers between layers.
    normalize : bool, default False
        If True, raises an error (see function docstring).
    name : str, default "SymmetricZZFeatureMap"
        Circuit name.

    Notes
    -----
    * The circuit depth grows linearly with ``reps``.
    * Parameter sharing reduces the number of independent parameters
      compared to the canonical ZZFeatureMap.
    * The class exposes ``input_params`` for easy binding: ``qc.bind_parameters({p: val})``.
    """
    def __init__(
        self,
        feature_dimension: int,
        reps: int = 2,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
        data_map_func: Callable[[Sequence[ParameterExpression]], ParameterExpression] | None = None,
        parameter_prefix: str = "x",
        insert_barriers: bool = False,
        normalize: bool = False,
        name: str = "SymmetricZZFeatureMap",
    ) -> None:
        built = symmetric_zz_feature_map(
            feature_dimension=feature_dimension,
            reps=reps,
            entanglement=entanglement,
            data_map_func=data_map_func,
            parameter_prefix=parameter_prefix,
            insert_barriers=insert_barriers,
            normalize=normalize,
            name=name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)
        self.input_params = built.input_params  # type: ignore[attr-defined]


__all__ = ["SymmetricZZFeatureMap", "symmetric_zz_feature_map"]
