"""ZZFeatureMapExtended – an extensible ZZ‑style feature map for Qiskit.

Features
--------
* **Higher‑order interactions** – support for 3‑body ZZ gates in addition
  to the standard pairwise ones.
* **RZZ entanglement** – optional use of the native :class:`~qiskit.circuit.library.RZZGate`
  for pairwise interactions.
* **Adaptive depth** – `reps` controls the number of repeated layers.
* **Pre‑rotations** – user‑supplied function can transform raw features
  before mapping.
* **Normalization helper** – utility to normalise a feature vector to
  unit Euclidean norm.
* **Compatibility** – exposes both a functional helper
  :func:`zz_feature_map_extended` and an OO subclass
  :class:`ZZFeatureMapExtended` that can be instantiated directly.

The module mirrors the original ``ZZFeatureMap`` API with additional
parameters and validation.  It is fully importable in any Qiskit
workflow that expects a :class:`~qiskit.circuit.QuantumCircuit` with
parameterised inputs.

Example
-------
>>> from zz_feature_map_extended import zz_feature_map_extended
>>> qc = zz_feature_map_extended(feature_dimension=4, reps=3,
...                              interaction_order=3, use_rzz=True,
...                              pre_rotation=lambda x: [xi*xi for xi in x])
>>> qc.draw()
"""

from __future__ import annotations

from math import pi
from typing import Callable, Iterable, List, Sequence, Tuple, Union

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterExpression, ParameterVector
from qiskit.circuit.library import RZZGate


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
      - ``"circular"``: linear plus wrap‑around (n‑1,0) if n > 2
      - explicit list of pairs like ``[(0, 2), (1, 3)]``
      - callable ``f(num_qubits) -> sequence of (i, j)``
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


def _default_map_1(x: ParameterExpression) -> ParameterExpression:
    """Default φ1(x) = x."""
    return x


def _default_map_2(x: ParameterExpression, y: ParameterExpression) -> ParameterExpression:
    """Default φ2(x, y) = (π − x)(π − y)."""
    return (pi - x) * (pi - y)


def _apply_zz_interaction(
    qc: QuantumCircuit,
    qubits: Iterable[int],
    angle: ParameterExpression,
    use_rzz: bool = False,
) -> None:
    """Apply a ZZ interaction between two qubits.

    If ``use_rzz`` is ``True``, a native :class:`~qiskit.circuit.library.RZZGate`
    is used; otherwise the CX–P–CX decomposition is employed.

    Parameters
    ----------
    qc : QuantumCircuit
        Target circuit.
    qubits : Iterable[int]
        Two qubit indices.
    angle : ParameterExpression
        Rotation angle (will be multiplied by 2 internally).
    use_rzz : bool
        Whether to use :class:`RZZGate`.
    """
    i, j = list(qubits)
    if use_rzz:
        qc.append(RZZGate(2 * angle), [i, j])
    else:
        qc.cx(i, j)
        qc.p(2 * angle, j)
        qc.cx(i, j)


def _apply_n_body_interaction(
    qc: QuantumCircuit,
    qubits: Iterable[int],
    angle: ParameterExpression,
) -> None:
    """Apply a 3‑body ZZ interaction via CX–P–CX decomposition.

    The sequence implements ``exp(-i * angle * Z_a Z_b Z_c / 2)``.
    """
    a, b, c = list(qubits)
    qc.cx(a, b)
    qc.cx(b, c)
    qc.p(2 * angle, c)
    qc.cx(b, c)
    qc.cx(a, b)


def _apply_pre_rotation(
    params: ParameterVector,
    pre_rot: Callable[[Sequence[ParameterExpression]], Sequence[ParameterExpression]] | None,
) -> List[ParameterExpression]:
    """Apply a user‑supplied pre‑rotation to the raw parameters.

    Returns a new list of parameters that will be used in the mapping functions.
    """
    if pre_rot is None:
        return list(params)
    transformed = pre_rot(list(params))
    if len(transformed)!= len(params):
        raise ValueError(
            f"pre_rotation returned {len(transformed)} parameters; expected {len(params)}."
        )
    return transformed


# ---------------------------------------------------------------------------
# Feature‑map builder
# ---------------------------------------------------------------------------

def zz_feature_map_extended(
    feature_dimension: int,
    reps: int = 2,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
    data_map_func: Callable[[Sequence[ParameterExpression]], ParameterExpression] | None = None,
    interaction_order: int = 2,
    use_rzz: bool = False,
    pre_rotation: Callable[[Sequence[ParameterExpression]], Sequence[ParameterExpression]] | None = None,
    normalize: bool = False,
    parameter_prefix: str = "x",
    insert_barriers: bool = False,
    name: str | None = None,
) -> QuantumCircuit:
    """Build an extended ZZ‑style feature map.

    Parameters
    ----------
    feature_dimension : int
        Number of classical features (also the number of qubits).
    reps : int
        Number of repeated layers.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]]
        Entanglement pattern for pairwise interactions.
    data_map_func : Callable[[Sequence[ParameterExpression]], ParameterExpression] | None
        Optional custom mapping from feature(s) to rotation angles.
    interaction_order : int
        2 for pairwise, 3 for 3‑body interactions.  Higher values are not supported.
    use_rzz : bool
        If ``True``, pairwise interactions use the native :class:`RZZGate`.
    pre_rotation : Callable[[Sequence[ParameterExpression]], Sequence[ParameterExpression]] | None
        Function applied to the raw feature parameters before mapping.
    normalize : bool
        If ``True``, the helper :func:`normalize_features` should be used to
        normalise the input data before binding.
    parameter_prefix : str
        Prefix for the :class:`ParameterVector` names.
    insert_barriers : bool
        Whether to insert barriers for readability.
    name : str | None
        Optional circuit name.

    Returns
    -------
    QuantumCircuit
        Parameterised circuit ready for binding.

    Raises
    ------
    ValueError
        If ``feature_dimension`` < 2, ``interaction_order`` not supported,
        or entanglement pairs are invalid.
    """
    if feature_dimension < 2:
        raise ValueError("feature_dimension must be >= 2 for ZZFeatureMapExtended.")

    if interaction_order not in (2, 3):
        raise ValueError("interaction_order must be 2 (pairwise) or 3 (three‑body).")

    n = int(feature_dimension)
    qc = QuantumCircuit(n, name=name or "ZZFeatureMapExtended")

    # Raw parameters
    raw_params = ParameterVector(parameter_prefix, n)
    # Apply optional pre‑rotation
    params = _apply_pre_rotation(raw_params, pre_rotation)

    # Choose mapping functions
    if data_map_func is None:
        map1 = _default_map_1
        map2 = _default_map_2
    else:
        def map1(xi: ParameterExpression) -> ParameterExpression:
            return data_map_func([xi])
        def map2(xi: ParameterExpression, xj: ParameterExpression) -> ParameterExpression:
            return data_map_func([xi, xj])

    pairs = _resolve_entanglement(n, entanglement)

    for rep in range(int(reps)):
        # Basis preparation
        qc.h(range(n))
        if insert_barriers:
            qc.barrier()

        # Single‑qubit phases
        for i in range(n):
            qc.p(2 * map1(params[i]), i)
        if insert_barriers:
            qc.barrier()

        # Pairwise or 3‑body interactions
        if interaction_order == 2:
            for (i, j) in pairs:
                angle_2 = 2 * map2(params[i], params[j])
                _apply_zz_interaction(qc, (i, j), angle_2, use_rzz=use_rzz)
        else:  # interaction_order == 3
            # Generate all unique 3‑tuples
            triples = [(i, j, k) for i in range(n) for j in range(i + 1, n) for k in range(j + 1, n)]
            for (i, j, k) in triples:
                angle_3 = 2 * map2(params[i], params[j])  # reuse map2 for simplicity
                # For 3‑body we could use a dedicated map3; here we approximate
                _apply_n_body_interaction(qc, (i, j, k), angle_3)

        if insert_barriers and rep!= reps - 1:
            qc.barrier()

    qc.input_params = raw_params  # type: ignore[attr-defined]
    return qc


# ---------------------------------------------------------------------------
# Class‑style wrapper
# ---------------------------------------------------------------------------

class ZZFeatureMapExtended(QuantumCircuit):
    """Class‑style wrapper for the extended ZZ‑feature map.

    Instantiation is equivalent to calling :func:`zz_feature_map_extended`
    with the same parameter set.  The resulting circuit inherits all
    methods of :class:`~qiskit.circuit.QuantumCircuit` and exposes
    ``input_params`` for binding.

    Example
    -------
    >>> from zz_feature_map_extended import ZZFeatureMapExtended
    >>> qc = ZZFeatureMapExtended(feature_dimension=5, reps=2,
   ...                            interaction_order=3, use_rzz=True)
    >>> qc.draw()
    """
    def __init__(
        self,
        feature_dimension: int,
        reps: int = 2,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
        data_map_func: Callable[[Sequence[ParameterExpression]], ParameterExpression] | None = None,
        interaction_order: int = 2,
        use_rzz: bool = False,
        pre_rotation: Callable[[Sequence[ParameterExpression]], Sequence[ParameterExpression]] | None = None,
        normalize: bool = False,
        parameter_prefix: str = "x",
        insert_barriers: bool = False,
        name: str = "ZZFeatureMapExtended",
    ) -> None:
        built = zz_feature_map_extended(
            feature_dimension=feature_dimension,
            reps=reps,
            entanglement=entanglement,
            data_map_func=data_map_func,
            interaction_order=interaction_order,
            use_rzz=use_rzz,
            pre_rotation=pre_rotation,
            normalize=normalize,
            parameter_prefix=parameter_prefix,
            insert_barriers=insert_barriers,
            name=name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)
        self.input_params = built.input_params  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def normalize_features(data: Sequence[float]) -> List[float]:
    """Return a unit‑norm copy of *data*.

    Parameters
    ----------
    data : Sequence[float]
        Raw feature vector.

    Returns
    -------
    List[float]
        Normalised feature vector with Euclidean norm 1.

    Raises
    ------
    ValueError
        If the input vector is zero.
    """
    import math
    norm = math.sqrt(sum(x * x for x in data))
    if norm == 0:
        raise ValueError("Cannot normalize a zero vector.")
    return [x / norm for x in data]


__all__ = [
    "ZZFeatureMapExtended",
    "zz_feature_map_extended",
    "normalize_features",
]
