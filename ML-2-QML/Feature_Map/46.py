"""Controlled‑symmetrised ZZ feature map with shared‑parameter scaling.

This module implements a variant of the polynomial ZZ feature map
originally defined in `zz_feature_map_poly.py`.  The key changes are:

- All single‑qubit phase terms are multiplied by a *shared scaling
  factor* (`shared_scale`).  This allows a global rescaling of the
  data‑dependent phases while keeping the individual parameter layout
  unchanged.

- The two‑qubit interaction term is replaced by a cosine‑based coupling
  `pair_weight * cos(x_i - x_j)`.  The cosine is an even function, so
  the coupling is automatically symmetrised with respect to the qubit
  indices.

- The rest of the circuit (basis preparation, entanglement pattern,
  repetition depth) remains identical to the original implementation.

The module exposes a functional interface
`zz_feature_map_poly_controlled(...)` and a Qiskit `QuantumCircuit`
subclass `ZZFeatureMapPolyControlled`.  Both can be used interchangeably
in standard Qiskit workflows.

Supported datasets
------------------
The circuit accepts a classical feature vector of length `feature_dimension`
and encodes it into a quantum state.  The mapping is fully parameterised
by the quantum circuit and can be used in variational algorithms or
quantum kernel methods.

Parameter constraints
---------------------
* `feature_dimension` must be an integer >= 2.
* `reps` must be a positive integer.
* `entanglement` can be a string (`"full"`, `"linear"`, `"circular"`) or
  a sequence of qubit pairs.
* `basis` may be `"h"` (Hadamard) or `"ry"` (RY(π/2)).
* `shared_scale` and `pair_weight` must be real numbers.

The function attaches the `ParameterVector` to the circuit as
`qc.input_params` for easy binding.

"""

from __future__ import annotations

from math import pi
from typing import Callable, List, Sequence, Tuple

import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterExpression, ParameterVector
from sympy import cos as sympy_cos


# ---------------------------------------------------------------------------
# Utility helpers
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
    # basic validation
    for (i, j) in pairs:
        if i == j:
            raise ValueError("Entanglement pairs must connect distinct qubits.")
        if not (0 <= i < num_qubits and 0 <= j < num_qubits):
            raise ValueError(f"Entanglement pair {(i, j)} out of range for n={num_qubits}.")
    return pairs


# ---------------------------------------------------------------------------
# Functional implementation
# ---------------------------------------------------------------------------

def zz_feature_map_poly_controlled(
    feature_dimension: int,
    reps: int = 2,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
    single_coeffs: Sequence[float] = (1.0,),
    pair_weight: float = 1.0,
    shared_scale: float = 1.0,
    basis: str = "h",
    parameter_prefix: str = "x",
    insert_barriers: bool = False,
    name: str | None = None,
) -> QuantumCircuit:
    """Controlled‑symmetrised polynomial ZZ feature map.

    Parameters
    ----------
    feature_dimension : int
        Number of qubits / length of the classical feature vector.
    reps : int, default 2
        Number of repetitions of the feature‑map block.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]]
        Entanglement pattern.  See :func:`_resolve_entanglement` for details.
    single_coeffs : Sequence[float], default (1.0,)
        Coefficients for the polynomial single‑qubit phase term.
    pair_weight : float, default 1.0
        Scaling factor for the two‑qubit coupling.
    shared_scale : float, default 1.0
        Global scaling applied to all single‑qubit phase terms.
    basis : str, default "h"
        Basis preparation: ``"h"`` for Hadamard, ``"ry"`` for RY(π/2).
    parameter_prefix : str, default "x"
        Prefix for the parameter vector names.
    insert_barriers : bool, default False
        Whether to insert barriers between logical sections.
    name : str | None, default None
        Optional circuit name.

    Returns
    -------
    QuantumCircuit
        Parameterised feature‑map circuit.
    """
    if feature_dimension < 2:
        raise ValueError("feature_dimension must be >= 2.")
    if reps < 1:
        raise ValueError("reps must be a positive integer.")
    n = int(feature_dimension)

    qc = QuantumCircuit(n, name=name or "ZZFeatureMapPolyControlled")

    # Parameter vector
    x = ParameterVector(parameter_prefix, n)

    # Define the single‑qubit phase map (allows polynomial terms)
    def map1(xi: ParameterExpression) -> ParameterExpression:
        expr: ParameterExpression = 0
        p = xi
        for c in single_coeffs:
            expr += c * p
            p *= xi
        return expr

    # Define the two‑qubit coupling map (cosine‑based, symmetric)
    def map2(xi: ParameterExpression, xj: ParameterExpression) -> ParameterExpression:
        return pair_weight * sympy_cos(xi - xj)

    pairs = _resolve_entanglement(n, entanglement)

    for rep in range(int(reps)):
        # Basis preparation
        if basis == "h":
            qc.h(range(n))
        elif basis == "ry":
            for q in range(n):
                qc.ry(pi / 2, q)
        else:
            raise ValueError("basis must be 'h' or 'ry'.")
        if insert_barriers:
            qc.barrier()

        # Single‑qubit phases
        for i in range(n):
            qc.p(2 * shared_scale * map1(x[i]), i)
        if insert_barriers:
            qc.barrier()

        # ZZ interaction via CX–P–CX
        for (i, j) in pairs:
            angle = 2 * map2(x[i], x[j])
            qc.cx(i, j)
            qc.p(angle, j)
            qc.cx(i, j)
        if insert_barriers and rep!= reps - 1:
            qc.barrier()

    qc.input_params = x  # type: ignore[attr-defined]
    return qc


# ---------------------------------------------------------------------------
# Class‑style wrapper
# ---------------------------------------------------------------------------

class ZZFeatureMapPolyControlled(QuantumCircuit):
    """Class‑style wrapper for the controlled‑symmetrised polynomial ZZ feature map.

    The constructor builds a :func:`zz_feature_map_poly_controlled` circuit
    and composes it into ``self``.  The resulting circuit inherits all
    methods of :class:`~qiskit.circuit.QuantumCircuit` and exposes
    ``self.input_params`` for easy parameter binding.
    """

    def __init__(
        self,
        feature_dimension: int,
        reps: int = 2,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
        single_coeffs: Sequence[float] = (1.0,),
        pair_weight: float = 1.0,
        shared_scale: float = 1.0,
        basis: str = "h",
        parameter_prefix: str = "x",
        insert_barriers: bool = False,
        name: str = "ZZFeatureMapPolyControlled",
    ) -> None:
        built = zz_feature_map_poly_controlled(
            feature_dimension,
            reps,
            entanglement,
            single_coeffs,
            pair_weight,
            shared_scale,
            basis,
            parameter_prefix,
            insert_barriers,
            name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)
        self.input_params = built.input_params  # type: ignore[attr-defined]


__all__ = ["ZZFeatureMapPolyControlled", "zz_feature_map_poly_controlled"]
