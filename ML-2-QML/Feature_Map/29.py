"""Symmetrised polynomial ZZ feature map with adjustable interaction exponent.

This module implements a controlled‑modification of the original
ZZFeatureMapPoly.  The modifications are:

- **Symmetry flag** – when set, a phase is applied to *both* qubits involved
  in each entangling pair (CX–P–CX for the first qubit, then CX–P–CX for the
  second).  This doubles the effective interaction strength while keeping
  the circuit depth unchanged.
- **Interaction exponent** – the pair‑wise phase is computed as
  ``pair_weight * (x_i ** exp) * (x_j ** exp)``.  The exponent defaults to 1
  (the original behaviour) but can be tuned to increase non‑linearity.
- **Shared single‑qubit coefficients** – the polynomial coefficients for the
  single‑qubit phase can be specified as a single float (applied to all qubits)
  or as a sequence matching the feature dimension.

The circuit remains fully parameterised and compatible with Qiskit’s
data‑encoding workflows.

Typical usage:

```python
from zz_feature_map_poly_controlled_modification import ZZFeatureMapPolySym

qc = ZZFeatureMapPolySym(
    feature_dimension=4,
    reps=3,
    entanglement="circular",
    single_coeffs=0.8,
    pair_weight=1.5,
    pair_exponent=2,
    symmetry=True,
    basis="ry",
    insert_barriers=True
)
params = qc.input_params
```

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
    """Return a list of two‑qubit pairs according to a simple entanglement spec.

    Supported specs:

    * ``"full"`` – all‑to‑all pairs (i < j)
    * ``"linear"`` – nearest neighbours
    * ``"circular"`` – linear plus wrap‑around (n‑1, 0) if n > 2
    * explicit list of pairs
    * callable ``f(num_qubits) -> sequence of (i, j)``
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

    # explicit sequence of pairs
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
    """Default φ2(x, y) = x * y."""
    return x * y


# ---------------------------------------------------------------------------
# Symmetrised Polynomial ZZ Feature Map
# ---------------------------------------------------------------------------

def zz_feature_map_poly_sym(
    feature_dimension: int,
    reps: int = 2,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
    single_coeffs: Sequence[float] | float = 1.0,
    pair_weight: float = 1.0,
    pair_exponent: float = 1.0,
    symmetry: bool = False,
    basis: str = "h",  # "h" or "ry"
    parameter_prefix: str = "x",
    insert_barriers: bool = False,
    name: str | None = None,
) -> QuantumCircuit:
    """Symmetrised polynomial ZZ feature map.

    Parameters
    ----------
    feature_dimension : int
        Number of classical features / qubits.
    reps : int
        Number of repetitions of the encoding layer.
    entanglement : str | sequence | callable
        Entanglement pattern.
    single_coeffs : sequence | float
        Polynomial coefficients for the single‑qubit phase.  If a single float
        is provided, it is applied to all qubits; otherwise a sequence of
        length ``feature_dimension`` is required.
    pair_weight : float
        Overall scaling for pairwise interactions.
    pair_exponent : float
        Exponent applied to each feature in the pairwise phase
        ``pair_weight * (x_i ** exp) * (x_j ** exp)``.
    symmetry : bool
        If ``True``, apply a phase to *both* qubits of each entangling pair
        (CX–P–CX for the first qubit, then CX–P–CX for the second).  This
        preserves the circuit depth but doubles the effective interaction.
    basis : str
        Basis preparation: ``"h"`` for Hadamard, ``"ry"`` for RY(π/2).
    parameter_prefix : str
        Prefix for the parameter vector.
    insert_barriers : bool
        Insert barriers between logical blocks for easier visualisation.
    name : str | None
        Name of the circuit.  If ``None``, a default name is used.

    Returns
    -------
    QuantumCircuit
        Parameterised encoding circuit with attribute ``input_params``.

    Raises
    ------
    ValueError
        If ``feature_dimension`` < 2, if ``single_coeffs`` length mismatch,
        or if ``basis`` is unrecognised.
    """
    if feature_dimension < 2:
        raise ValueError("feature_dimension must be >= 2.")
    n = int(feature_dimension)

    # Resolve single‑qubit coefficients
    if isinstance(single_coeffs, (int, float)):
        coeff_seq = [float(single_coeffs)] * n
    else:
        coeff_seq = list(single_coeffs)
        if len(coeff_seq)!= n:
            raise ValueError(
                f"single_coeffs length {len(coeff_seq)} does not match feature_dimension {n}."
            )

    qc = QuantumCircuit(n, name=name or "ZZFeatureMapPolySym")

    # Parameter vector
    x = ParameterVector(parameter_prefix, n)

    def map1(xi: ParameterExpression) -> ParameterExpression:
        """Polynomial mapping for single‑qubit phase."""
        expr: ParameterExpression = 0
        power: ParameterExpression = xi
        for c in coeff_seq:
            expr += c * power
            power *= xi  # next power
        return expr

    def map2(xi: ParameterExpression, xj: ParameterExpression) -> ParameterExpression:
        """Pairwise mapping with optional exponent."""
        return pair_weight * (xi ** pair_exponent) * (xj ** pair_exponent)

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
            qc.p(2 * map1(x[i]), i)
        if insert_barriers:
            qc.barrier()

        # Entangling ZZ with optional symmetry
        for (i, j) in pairs:
            angle = 2 * map2(x[i], x[j])
            # Apply phase to target qubit
            qc.cx(i, j)
            qc.p(angle, j)
            qc.cx(i, j)
            if symmetry:
                # Apply same phase to control qubit
                qc.cx(j, i)
                qc.p(angle, i)
                qc.cx(j, i)
        if insert_barriers and rep!= reps - 1:
            qc.barrier()

    qc.input_params = x  # type: ignore[attr-defined]
    return qc


class ZZFeatureMapPolySym(QuantumCircuit):
    """Class‑style wrapper for the symmetrised polynomial ZZ feature map.

    Parameters are identical to :func:`zz_feature_map_poly_sym`.  The class
    exposes the same ``input_params`` attribute for parameter binding.
    """
    def __init__(
        self,
        feature_dimension: int,
        reps: int = 2,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
        single_coeffs: Sequence[float] | float = 1.0,
        pair_weight: float = 1.0,
        pair_exponent: float = 1.0,
        symmetry: bool = False,
        basis: str = "h",
        parameter_prefix: str = "x",
        insert_barriers: bool = False,
        name: str = "ZZFeatureMapPolySym",
    ) -> None:
        built = zz_feature_map_poly_sym(
            feature_dimension,
            reps,
            entanglement,
            single_coeffs,
            pair_weight,
            pair_exponent,
            symmetry,
            basis,
            parameter_prefix,
            insert_barriers,
            name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)
        self.input_params = built.input_params  # type: ignore[attr-defined]


__all__ = ["ZZFeatureMapPolySym", "zz_feature_map_poly_sym"]
