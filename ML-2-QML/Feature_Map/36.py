"""Symmetrised ZZ feature map with shared parameters and optional scaling.

This module implements a controlled‑modification of the original
`ZZFeatureMapPoly`.  The changes introduced are:

* **Symmetric entanglement** – an optional ``symmetric`` entanglement
  mode that couples each qubit with its opposite on the ring.
* **Parameter sharing** – a ``share_params`` flag that forces all
  repetitions to use the same parameter vector, reducing the number of
  free parameters.
* **Feature scaling** – a ``scaling_factor`` that multiplies the input
  features before they are mapped.  This is useful when the data is
  not naturally in the ``[-π,π]`` range.
* **Pre/post rotations** – optional callables that can be used to
  prepend or append custom gates to each repetition.

Both a functional helper ``zz_feature_map_poly_sym`` and a
``QuantumCircuit`` subclass ``ZZFeatureMapPolySym`` are provided.

The API is intentionally compatible with Qiskit's data‑encoding
workflows: the returned circuit exposes an ``input_params`` attribute
containing the parameters that must be bound.

Typical usage:

```python
from zz_feature_map_poly_sym import zz_feature_map_poly_sym
qc = zz_feature_map_poly_sym(feature_dimension=4, reps=2)
qc.bind_parameters({p: 0.5 for p in qc.input_params})
```

"""

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
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]],
) -> List[Tuple[int, int]]:
    """Return a list of two‑qubit pairs according to a simple entanglement spec.

    Supported specs:
      - ``"full"``: all‑to‑all pairs (i < j)
      - ``"linear"``: nearest neighbours (0,1), (1,2), …
      - ``"circular"``: linear plus wrap‑around (n‑1,0) if n > 2
      - ``"symmetric"``: each qubit coupled to its opposite on the ring
      - explicit list of pairs like [(0, 2), (1, 3)]
      - callable: f(num_qubits) -> sequence of (i, j)

    Raises
    ------
    ValueError
        If the spec is unknown or contains invalid pairs.
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
        if entanglement == "symmetric":
            half = num_qubits // 2
            return [(i, num_qubits - i - 1) for i in range(half)]
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


# ---------------------------------------------------------------------------
# Feature map
# ---------------------------------------------------------------------------

def zz_feature_map_poly_sym(
    feature_dimension: int,
    reps: int = 2,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
    single_coeffs: Sequence[float] = (1.0,),
    pair_weight: float = 1.0,
    basis: str = "h",
    parameter_prefix: str = "x",
    scaling_factor: float = 1.0,
    share_params: bool = True,
    insert_barriers: bool = False,
    pre_rotation: Callable[[QuantumCircuit, Sequence[int]], None] | None = None,
    post_rotation: Callable[[QuantumCircuit, Sequence[int]], None] | None = None,
    name: str | None = None,
) -> QuantumCircuit:
    """Symmetrised ZZ feature map with optional scaling and parameter sharing.

    Parameters
    ----------
    feature_dimension : int
        Number of qubits / features. Must be >= 2.
    reps : int, default 2
        Number of repetitions of the feature‑map block.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]], default "full"
        Entanglement specification. ``"symmetric"`` couples each qubit with its opposite.
    single_coeffs : Sequence[float], default (1.0,)
        Coefficients for the single‑qubit polynomial map.
    pair_weight : float, default 1.0
        Weight for the two‑qubit ZZ interaction.
    basis : str, default "h"
        Basis preparation: ``"h"`` for Hadamard or ``"ry"`` for RY(π/2).
    parameter_prefix : str, default "x"
        Prefix for the parameter vector names.
    scaling_factor : float, default 1.0
        Multiplicative scaling applied to all feature values before mapping.
    share_params : bool, default True
        If ``True``, all repetitions use the same ParameterVector.
    insert_barriers : bool, default False
        If ``True``, barrier gates are inserted between major blocks.
    pre_rotation : Callable[[QuantumCircuit, Sequence[int]], None] | None, default None
        Optional function applied before each basis preparation.
    post_rotation : Callable[[QuantumCircuit, Sequence[int]], None] | None, default None
        Optional function applied after the ZZ interactions.
    name : str | None, default None
        Name for the resulting circuit.

    Returns
    -------
    QuantumCircuit
        The constructed feature‑map circuit.  The circuit exposes an
        ``input_params`` attribute that holds the ParameterVector(s).

    Raises
    ------
    ValueError
        If input arguments are invalid.
    """
    if feature_dimension < 2:
        raise ValueError("feature_dimension must be >= 2.")
    if reps < 1:
        raise ValueError("reps must be >= 1.")
    if pair_weight < 0:
        raise ValueError("pair_weight must be non‑negative.")
    if scaling_factor <= 0:
        raise ValueError("scaling_factor must be positive.")
    if not single_coeffs:
        raise ValueError("single_coeffs must contain at least one coefficient.")
    if basis not in ("h", "ry"):
        raise ValueError("basis must be 'h' or 'ry'.")

    n = int(feature_dimension)
    qc = QuantumCircuit(n, name=name or "ZZFeatureMapPolySym")

    # Create parameter vector(s)
    if share_params:
        x = ParameterVector(parameter_prefix, n)
        param_vectors = [x]
    else:
        param_vectors = [ParameterVector(f"{parameter_prefix}_{rep}", n) for rep in range(reps)]

    def map1(xi: ParameterExpression) -> ParameterExpression:
        """Polynomial single‑qubit map."""
        expr: ParameterExpression = 0
        p = xi
        for coeff in single_coeffs:
            expr += coeff * p
            p = p * xi  # next power
        return scaling_factor * expr

    def map2(xi: ParameterExpression, xj: ParameterExpression) -> ParameterExpression:
        """Two‑qubit interaction map."""
        return pair_weight * (xi + xj)

    pairs = _resolve_entanglement(n, entanglement)

    for rep in range(int(reps)):
        current_params = param_vectors[0] if share_params else param_vectors[rep]

        # Optional pre‑rotation
        if pre_rotation is not None:
            pre_rotation(qc, range(n))

        # Basis preparation
        if basis == "h":
            qc.h(range(n))
        else:
            for q in range(n):
                qc.ry(pi / 2, q)

        if insert_barriers:
            qc.barrier()

        # Single‑qubit phases
        for i in range(n):
            qc.p(2 * map1(current_params[i]), i)

        if insert_barriers:
            qc.barrier()

        # ZZ interaction via CX–P–CX
        for (i, j) in pairs:
            angle = 2 * map2(current_params[i], current_params[j])
            qc.cx(i, j)
            qc.p(angle, j)
            qc.cx(i, j)

        # Optional post‑rotation
        if post_rotation is not None:
            post_rotation(qc, range(n))

        if insert_barriers and rep!= reps - 1:
            qc.barrier()

    # Attach parameters for binding
    qc.input_params = param_vectors[0] if share_params else param_vectors

    return qc


class ZZFeatureMapPolySym(QuantumCircuit):
    """Class‑style wrapper for the symmetrised polynomial ZZ feature map.

    Parameters
    ----------
    feature_dimension : int
        Number of qubits / features. Must be >= 2.
    reps : int, default 2
        Number of repetitions of the feature‑map block.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]], default "full"
        Entanglement specification. ``"symmetric"`` couples each qubit with its opposite.
    single_coeffs : Sequence[float], default (1.0,)
        Coefficients for the single‑qubit polynomial map.
    pair_weight : float, default 1.0
        Weight for the two‑qubit ZZ interaction.
    basis : str, default "h"
        Basis preparation: ``"h"`` for Hadamard or ``"ry"`` for RY(π/2).
    parameter_prefix : str, default "x"
        Prefix for the parameter vector names.
    scaling_factor : float, default 1.0
        Multiplicative scaling applied to all feature values before mapping.
    share_params : bool, default True
        If ``True``, all repetitions use the same ParameterVector.
    insert_barriers : bool, default False
        If ``True``, barrier gates are inserted between major blocks.
    pre_rotation : Callable[[QuantumCircuit, Sequence[int]], None] | None, default None
        Optional function applied before each basis preparation.
    post_rotation : Callable[[QuantumCircuit, Sequence[int]], None] | None, default None
        Optional function applied after the ZZ interactions.
    name : str, default "ZZFeatureMapPolySym"
        Name for the circuit.
    """

    def __init__(
        self,
        feature_dimension: int,
        reps: int = 2,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
        single_coeffs: Sequence[float] = (1.0,),
        pair_weight: float = 1.0,
        basis: str = "h",
        parameter_prefix: str = "x",
        scaling_factor: float = 1.0,
        share_params: bool = True,
        insert_barriers: bool = False,
        pre_rotation: Callable[[QuantumCircuit, Sequence[int]], None] | None = None,
        post_rotation: Callable[[QuantumCircuit, Sequence[int]], None] | None = None,
        name: str = "ZZFeatureMapPolySym",
    ) -> None:
        built = zz_feature_map_poly_sym(
            feature_dimension,
            reps,
            entanglement,
            single_coeffs,
            pair_weight,
            basis,
            parameter_prefix,
            scaling_factor,
            share_params,
            insert_barriers,
            pre_rotation,
            post_rotation,
            name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)
        self.input_params = built.input_params  # type: ignore[attr-defined]


__all__ = ["ZZFeatureMapPolySym", "zz_feature_map_poly_sym"]
