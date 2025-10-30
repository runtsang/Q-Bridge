"""ZZFeatureMapPolyControlled
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
A controlled‑modification variant of the polynomial ZZ feature map.

Features
--------
- **Distance‑weighted pair interactions**: pair weights modulated by the
  (normalised) distance between qubits.
- **Per‑qubit coefficient vectors**: when ``coefficients_shared`` is ``False``,
  each qubit may have its own polynomial coefficient list.
- **Data re‑parameterisation**: a user‑supplied callable can transform each
  feature before it enters the map.
- **Feature scaling**: a scalar parameter multiplies all input features.
- **Barrier support**: optional barriers between logical blocks.
- **Basis choice**: Hadamard or RY(π/2) preparation.

The module exposes both a functional helper (`zz_feature_map_poly_controlled`)
and an OO wrapper (`ZZFeatureMapPolyControlled`).

Examples
--------
>>> from zz_feature_map_poly_controlled_modification import zz_feature_map_poly_controlled
>>> qc = zz_feature_map_poly_controlled(feature_dimension=4, reps=2, distance_weighting=True)
>>> qc.draw()
"""
from __future__ import annotations

from math import pi
from typing import Callable, Sequence, Tuple, List, Union, Optional

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
      - ``"circular"``: linear plus wrap‑around (n‑1,0) if n > 2
      - explicit list of pairs like [(0, 2), (1, 3)]
      - callable: ``f(num_qubits) -> sequence of (i, j)``
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
    """Default φ₁(x) = x."""
    return x


def _default_map_2(x: ParameterExpression, y: ParameterExpression) -> ParameterExpression:
    """Default φ₂(x, y) = (π − x)(π − y)."""
    return (pi - x) * (pi - y)


# ---------------------------------------------------------------------------
# Main feature‑map construction
# ---------------------------------------------------------------------------

def zz_feature_map_poly_controlled(
    feature_dimension: int,
    reps: int = 1,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
    single_coeffs: Sequence[float] | Sequence[Sequence[float]] = (1.0,),
    coefficients_shared: bool = True,
    pair_weight: float | ParameterExpression = 1.0,
    distance_weighting: bool = False,
    reparam_func: Callable[[ParameterExpression], ParameterExpression] = lambda x: x,
    scaling_factor: float | ParameterExpression = 1.0,
    basis: str = "h",
    insert_barriers: bool = False,
    name: str | None = None,
    parameter_prefix: str = "x",
) -> QuantumCircuit:
    """
    Construct a distance‑weighted polynomial ZZ feature map.

    Parameters
    ----------
    feature_dimension : int
        Number of qubits / dimension of the input feature vector.
    reps : int, default 1
        Number of feature‑map repetitions.
    entanglement : str | Sequence | Callable, default "full"
        Entanglement pattern between qubits.
    single_coeffs : Sequence[float] | Sequence[Sequence[float]], default (1.0,)
        Polynomial coefficients for φ₁(x).  If ``coefficients_shared`` is ``True``,
        a single list is used for all qubits; otherwise a list of lists is expected
        with length equal to ``feature_dimension``.
    coefficients_shared : bool, default True
        Whether the same coefficient vector is used for all qubits.
    pair_weight : float | ParameterExpression, default 1.0
        Base weight for the pair interaction φ₂(x, y) = pair_weight · x · y.
    distance_weighting : bool, default False
        If ``True``, the pair weight is modulated by the (normalised) distance
        between qubits: weight = pair_weight · (1 – d/(n‑1)).
    reparam_func : Callable[[ParameterExpression], ParameterExpression], default identity
        Function applied to each (scaled) feature before it enters the map.
    scaling_factor : float | ParameterExpression, default 1.0
        Scalar multiplier applied to every feature before re‑parameterisation.
    basis : str, default "h"
        Basis preparation: ``"h"`` for Hadamard, ``"ry"`` for RY(π/2).
    insert_barriers : bool, default False
        Whether to insert barriers between logical blocks.
    name : str | None, default None
        Name of the resulting circuit.
    parameter_prefix : str, default "x"
        Prefix for the single‑qubit parameter vector.

    Returns
    -------
    QuantumCircuit
        The constructed feature‑map circuit.  The circuit exposes an
        ``input_params`` attribute containing the parameter vector.

    Raises
    ------
    ValueError
        If input arguments are inconsistent or out of bounds.
    """
    if feature_dimension < 2:
        raise ValueError("feature_dimension must be >= 2.")
    if reps < 1:
        raise ValueError("reps must be >= 1.")
    if not callable(reparam_func):
        raise ValueError("reparam_func must be callable.")
    if not isinstance(scaling_factor, (int, float, ParameterExpression)):
        raise ValueError("scaling_factor must be numeric or ParameterExpression.")
    if not isinstance(pair_weight, (int, float, ParameterExpression)):
        raise ValueError("pair_weight must be numeric or ParameterExpression.")
    if not isinstance(basis, str) or basis not in ("h", "ry"):
        raise ValueError("basis must be 'h' or 'ry'.")

    n = int(feature_dimension)

    if not coefficients_shared:
        if len(single_coeffs)!= n:
            raise ValueError("When coefficients_shared is False, single_coeffs must have length equal to feature_dimension.")
        for idx, coeff in enumerate(single_coeffs):
            if not isinstance(coeff, Sequence):
                raise ValueError(f"Coefficient list for qubit {idx} must be a sequence.")

    qc = QuantumCircuit(n, name=name or "ZZFeatureMapPolyControlled")

    # Parameter vector for single‑qubit data
    x = ParameterVector(parameter_prefix, n)

    # Resolve entanglement pairs
    pairs = _resolve_entanglement(n, entanglement)

    # Helper to compute φ₁ for a given qubit
    def _phi1(xi: ParameterExpression, coeffs: Sequence[float]) -> ParameterExpression:
        expr: ParameterExpression = 0
        power: ParameterExpression = xi
        for c in coeffs:
            expr = expr + c * power
            power = power * xi
        return expr

    # Main loop over repetitions
    for rep in range(int(reps)):
        # Basis preparation
        if basis == "h":
            qc.h(range(n))
        else:  # basis == "ry"
            for q in range(n):
                qc.ry(pi / 2, q)

        if insert_barriers:
            qc.barrier()

        # Single‑qubit phases
        for i in range(n):
            coeffs = single_coeffs if coefficients_shared else single_coeffs[i]
            xi_scaled = reparam_func(x[i] * scaling_factor)
            phi1 = _phi1(xi_scaled, coeffs)
            qc.p(2 * phi1, i)

        if insert_barriers:
            qc.barrier()

        # Two‑qubit ZZ interactions
        for (i, j) in pairs:
            xi_scaled = reparam_func(x[i] * scaling_factor)
            xj_scaled = reparam_func(x[j] * scaling_factor)
            distance = abs(i - j)
            dist_factor: ParameterExpression = 1
            if distance_weighting:
                dist_factor = 1 - distance / (n - 1)
            angle = 2 * pair_weight * xi_scaled * xj_scaled * dist_factor
            qc.cx(i, j)
            qc.p(angle, j)
            qc.cx(i, j)

        if insert_barriers and rep!= reps - 1:
            qc.barrier()

    # Expose the parameter vector for user binding
    qc.input_params = x  # type: ignore[attr-defined]
    return qc


# ---------------------------------------------------------------------------
# OO Wrapper
# ---------------------------------------------------------------------------

class ZZFeatureMapPolyControlled(QuantumCircuit):
    """
    OO wrapper for ``zz_feature_map_poly_controlled``.

    Parameters
    ----------
    Same as :func:`zz_feature_map_poly_controlled`.

    Notes
    -----
    The wrapper composes the built circuit into itself and forwards the
    ``input_params`` attribute for convenient parameter binding.
    """
    def __init__(
        self,
        feature_dimension: int,
        reps: int = 1,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
        single_coeffs: Sequence[float] | Sequence[Sequence[float]] = (1.0,),
        coefficients_shared: bool = True,
        pair_weight: float | ParameterExpression = 1.0,
        distance_weighting: bool = False,
        reparam_func: Callable[[ParameterExpression], ParameterExpression] = lambda x: x,
        scaling_factor: float | ParameterExpression = 1.0,
        basis: str = "h",
        insert_barriers: bool = False,
        name: str = "ZZFeatureMapPolyControlled",
        parameter_prefix: str = "x",
    ) -> None:
        built = zz_feature_map_poly_controlled(
            feature_dimension,
            reps,
            entanglement,
            single_coeffs,
            coefficients_shared,
            pair_weight,
            distance_weighting,
            reparam_func,
            scaling_factor,
            basis,
            insert_barriers,
            name,
            parameter_prefix,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)
        self.input_params = built.input_params  # type: ignore[attr-defined]


__all__ = ["ZZFeatureMapPolyControlled", "zz_feature_map_poly_controlled"]
