"""Controlled‑Modification Polynomial ZZ Feature Map.

This module implements a Qiskit‑compatible feature map that builds upon the
standard polynomial ZZ map but adds several controlled‑modification
features:

* **Shared parameters** – All qubits can share the same polynomial
  coefficients (`single_coeffs`) or each qubit can have its own list
  (`single_coeffs_per_qubit`).
* **Data rescaling** – A global `data_rescale` factor is applied to all
  classical features before they enter the map.
* **Optional per‑pair weights** – A uniform `pair_weight` applies to all
  ZZ couplings, or a custom mapping (`pair_weights`) can be supplied.
* **Flexible basis preparation** – Choose between Hadamard, RY(π/2), or
  no basis rotation.
* **Entanglement control** – Any of the usual entanglement specifications
  (`"full"`, `"linear"`, `"circular"`, explicit list, or callable) is
  accepted.

Both a functional helper (`zz_feature_map_poly_controlled`) and a
`QuantumCircuit` subclass (`ZZFeatureMapPolyControlled`) are provided
for convenient use.

The circuit supports standard Qiskit parameter binding and can be
directly integrated into VQE, QAOA, or variational quantum classifiers.

Example
-------
>>> from zz_feature_map_poly_controlled_modification import zz_feature_map_poly_controlled
>>> qc = zz_feature_map_poly_controlled(
...     feature_dimension=4,
...     reps=3,
...     single_coeffs=(1.0, 0.5),
...     data_rescale=0.8,
...     basis='ry',
...     entanglement='linear',
... )
>>> print(qc.draw())
"""

from __future__ import annotations

from math import pi
from typing import Callable, Mapping, Sequence, Tuple, List, Optional

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
      - ``"linear"``: nearest neighbors (0,1), (1,2), …
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
# Feature map
# ---------------------------------------------------------------------------

def zz_feature_map_poly_controlled(
    feature_dimension: int,
    reps: int = 2,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
    single_coeffs: Sequence[float] = (1.0,),
    single_coeffs_per_qubit: Sequence[Sequence[float]] | None = None,
    pair_weight: float = 1.0,
    pair_weights: Mapping[Tuple[int, int], float] | None = None,
    basis: str = "h",  # "h", "ry", or "none"
    data_rescale: float = 1.0,
    insert_barriers: bool = False,
    parameter_prefix: str = "x",
    name: str | None = None,
) -> QuantumCircuit:
    """
    Polynomial ZZ feature map with shared‑parameter symmetries.

    Parameters
    ----------
    feature_dimension : int
        Number of qubits / dimensionality of the classical feature vector.
    reps : int, default=2
        Number of repetitions of the feature‑map pattern.
    entanglement : str | list | callable, default="full"
        Specification of two‑qubit entanglement pairs.
    single_coeffs : sequence of float, default=(1.0,)
        Polynomial coefficients for the single‑qubit phase map.
        If ``single_coeffs_per_qubit`` is ``None``, the same list is applied
        to all qubits.
    single_coeffs_per_qubit : sequence of sequence of float, optional
        Per‑qubit polynomial coefficient lists.  Must have length equal to
        ``feature_dimension``.  Overrides ``single_coeffs`` for that qubit.
    pair_weight : float, default=1.0
        Global weight for the ZZ interaction term.
    pair_weights : mapping of (i, j) -> float, optional
        Custom weight for each ZZ pair.  If provided, overrides ``pair_weight``
        for the specified pairs.
    basis : str, default="h"
        Basis preparation before each repetition.
        Options: ``"h"`` (Hadamard), ``"ry"`` (RY(π/2)), ``"none"``
        (no basis rotation).
    data_rescale : float, default=1.0
        Global scaling factor applied to all classical features.
    insert_barriers : bool, default=False
        Insert barriers between layers for visual clarity.
    parameter_prefix : str, default="x"
        Prefix for the parameter vector.
    name : str, optional
        Name of the resulting quantum circuit.

    Returns
    -------
    QuantumCircuit
        Parameterised feature‑map circuit ready for binding.

    Notes
    -----
    * The circuit depth scales linearly with ``reps``.
    * The mapping functions are:

      * ``φ₁(x) = Σ_k c_k · x^{k+1}``  
        where ``c_k`` are the polynomial coefficients.
      * ``φ₂(x, y) = w · x · y``  
        where ``w`` is the pair weight.

    * All classical features are first multiplied by ``data_rescale`` before
      entering the map.
    """
    if feature_dimension < 2:
        raise ValueError("feature_dimension must be >= 2.")
    if reps < 1:
        raise ValueError("reps must be a positive integer.")
    if data_rescale == 0.0:
        raise ValueError("data_rescale cannot be zero; it would erase all data.")

    n = int(feature_dimension)
    qc = QuantumCircuit(n, name=name or "ZZFeatureMapPolyControlled")

    # Parameter vector for classical data
    x = ParameterVector(parameter_prefix, n)

    # Resolve entanglement pairs
    pairs = _resolve_entanglement(n, entanglement)

    # Helper: compute φ₁ for a single qubit
    def _phi1(xi: ParameterExpression, coeffs: Sequence[float]) -> ParameterExpression:
        expr: ParameterExpression = 0
        power = xi  # x^(1)
        for c in coeffs:
            expr = expr + c * power
            power = power * xi  # next power
        return expr

    # Helper: compute φ₂ for a pair
    def _phi2(xi: ParameterExpression, xj: ParameterExpression, weight: float) -> ParameterExpression:
        return weight * xi * xj

    for rep in range(int(reps)):
        # Basis preparation
        if basis == "h":
            qc.h(range(n))
        elif basis == "ry":
            for q in range(n):
                qc.ry(pi / 2, q)
        elif basis == "none":
            pass
        else:
            raise ValueError("basis must be 'h', 'ry', or 'none'.")

        if insert_barriers:
            qc.barrier()

        # Single‑qubit phases φ₁
        for i in range(n):
            coeffs = single_coeffs_per_qubit[i] if single_coeffs_per_qubit is not None else single_coeffs
            angle = 2 * _phi1(data_rescale * x[i], coeffs)
            qc.p(angle, i)

        if insert_barriers:
            qc.barrier()

        # ZZ interactions φ₂ via CX–P–CX
        for (i, j) in pairs:
            weight = pair_weights.get((i, j), pair_weight) if pair_weights else pair_weight
            angle = 2 * _phi2(data_rescale * x[i], data_rescale * x[j], weight)
            qc.cx(i, j)
            qc.p(angle, j)
            qc.cx(i, j)

        if insert_barriers and rep!= reps - 1:
            qc.barrier()

    qc.input_params = x  # type: ignore[attr-defined]
    return qc


class ZZFeatureMapPolyControlled(QuantumCircuit):
    """
    OO wrapper for the polynomial ZZ feature map with controlled‑modification.

    Parameters
    ----------
    feature_dimension : int
        Number of qubits / dimensionality of the classical feature vector.
    reps : int, default=2
        Number of repetitions of the feature‑map pattern.
    entanglement : str | list | callable, default="full"
        Specification of two‑qubit entanglement pairs.
    single_coeffs : sequence of float, default=(1.0,)
        Polynomial coefficients for the single‑qubit phase map.
    single_coeffs_per_qubit : sequence of sequence of float, optional
        Per‑qubit polynomial coefficient lists.
    pair_weight : float, default=1.0
        Global weight for the ZZ interaction term.
    pair_weights : mapping of (i, j) -> float, optional
        Custom weight for each ZZ pair.
    basis : str, default="h"
        Basis preparation before each repetition.
    data_rescale : float, default=1.0
        Global scaling factor applied to all classical features.
    insert_barriers : bool, default=False
        Insert barriers between layers for visual clarity.
    parameter_prefix : str, default="x"
        Prefix for the parameter vector.
    name : str, default="ZZFeatureMapPolyControlled"
        Name of the resulting quantum circuit.

    Notes
    -----
    The class behaves like a normal ``QuantumCircuit`` but exposes the
    ``input_params`` attribute for convenient parameter binding.
    """

    def __init__(
        self,
        feature_dimension: int,
        reps: int = 2,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
        single_coeffs: Sequence[float] = (1.0,),
        single_coeffs_per_qubit: Sequence[Sequence[float]] | None = None,
        pair_weight: float = 1.0,
        pair_weights: Mapping[Tuple[int, int], float] | None = None,
        basis: str = "h",
        data_rescale: float = 1.0,
        insert_barriers: bool = False,
        parameter_prefix: str = "x",
        name: str = "ZZFeatureMapPolyControlled",
    ) -> None:
        built = zz_feature_map_poly_controlled(
            feature_dimension,
            reps,
            entanglement,
            single_coeffs,
            single_coeffs_per_qubit,
            pair_weight,
            pair_weights,
            basis,
            data_rescale,
            insert_barriers,
            parameter_prefix,
            name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)
        self.input_params = built.input_params  # type: ignore[attr-defined]


__all__ = ["ZZFeatureMapPolyControlled", "zz_feature_map_poly_controlled"]
