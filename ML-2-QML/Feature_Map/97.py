"""Controlled‑modification polynomial ZZ feature map for Qiskit.

This module implements a Qiskit‑compatible feature map that extends the
original polynomial ZZ map with:
  • Optional feature‑vector normalisation (1/√n scaling).
  • Per‑qubit or shared polynomial coefficients.
  • Optional pre‑ and post‑rotations (RY(π/2)) that sandwich the main
    entangling layers.
  • Explicit error handling for inconsistent parameter shapes.

Both a functional helper (`zz_feature_map_poly_controlled_modification`) and
an OO wrapper (`ZZFeatureMapPolyControlledModification`) are provided
so users can choose the style that best fits their workflow.
"""

from __future__ import annotations

from math import pi, sqrt
from typing import Callable, Sequence, Tuple

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector, ParameterExpression


# ---------------------------------------------------------------------------
# Entanglement utilities
# ---------------------------------------------------------------------------

def _resolve_entanglement(
    num_qubits: int,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]],
) -> list[Tuple[int, int]]:
    """
    Resolve an entanglement specification into a list of two‑qubit pairs.

    Supported specifications:
      * ``"full"`` – all‑to‑all pairs (i < j)
      * ``"linear"`` – nearest‑neighbour chain
      * ``"circular"`` – linear chain with a wrap‑around connection
      * ``"full_sym"`` – synonym for ``"full"`` (kept for backward compatibility)
      * explicit list of tuples
      * callable ``f(num_qubits)`` returning a sequence of pairs

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    entanglement : str | Sequence[Tuple[int, int]] | Callable
        Entanglement specification.

    Returns
    -------
    list[Tuple[int, int]]
        Validated list of qubit pairs.

    Raises
    ------
    ValueError
        If the specification is unknown or contains invalid pairs.
    """
    if isinstance(entanglement, str):
        if entanglement in {"full", "full_sym"}:
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
    for (i, j) in pairs:
        if i == j:
            raise ValueError("Entanglement pairs must connect distinct qubits.")
        if not (0 <= i < num_qubits and 0 <= j < num_qubits):
            raise ValueError(f"Entanglement pair {(i, j)} out of range for n={num_qubits}.")
    return pairs


# ---------------------------------------------------------------------------
# Feature‑map construction
# ---------------------------------------------------------------------------

def zz_feature_map_poly_controlled_modification(
    feature_dimension: int,
    reps: int = 2,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
    single_coeffs: Sequence[float] | Sequence[Sequence[float]] = (1.0,),
    pair_weight: float = 1.0,
    basis: str = "h",
    parameter_prefix: str = "x",
    insert_barriers: bool = False,
    normalize: bool = False,
    shared_coeffs: bool = True,
    apply_prepost_rotations: bool = False,
    name: str | None = None,
) -> QuantumCircuit:
    """
    Construct a polynomial ZZ feature map with controlled modifications.

    The map encodes classical data ``x`` into a quantum circuit by applying
    single‑qubit phases determined by a polynomial of the feature values
    and two‑qubit ZZ interactions that depend on the product of feature pairs.

    Modifications relative to the original seed:
      * **Normalization** – when ``normalize=True`` each feature is divided by
        ``sqrt(n)`` before entering the map, improving numerical stability.
      * **Shared vs. per‑qubit coefficients** – ``shared_coeffs`` toggles
        whether all qubits use the same polynomial coefficient list or each
        qubit receives its own list.
      * **Pre/post rotations** – optional RY(π/2) rotations applied before the
        first repetition and after the last, providing an additional layer
        of basis change without altering the core entanglement structure.

    Parameters
    ----------
    feature_dimension : int
        Number of classical features / qubits.
    reps : int, default 2
        Number of repetitions of the entangling block.
    entanglement : str | Sequence[Tuple[int, int]] | Callable
        Entanglement topology.
    single_coeffs : Sequence[float] | Sequence[Sequence[float]]
        Polynomial coefficients. If ``shared_coeffs`` is True, this is a
        single sequence applied to all qubits; otherwise it must be a list
        of sequences, one per qubit.
    pair_weight : float, default 1.0
        Scaling factor for the two‑qubit ZZ terms.
    basis : str, default "h"
        Basis preparation: ``"h"`` for Hadamard, ``"ry"`` for RY(π/2).
    parameter_prefix : str, default "x"
        Prefix used for the parameter vector.
    insert_barriers : bool, default False
        Insert barriers after each logical block for visual clarity.
    normalize : bool, default False
        If True, scale each feature by ``1/√n`` before encoding.
    shared_coeffs : bool, default True
        If False, ``single_coeffs`` must be a list of coefficient sequences
        matching ``feature_dimension``.
    apply_prepost_rotations : bool, default False
        If True, apply RY(π/2) rotations before the first repetition and
        after the last repetition.
    name : str | None, default None
        Optional name for the resulting circuit.

    Returns
    -------
    QuantumCircuit
        The constructed feature‑map circuit.

    Raises
    ------
    ValueError
        For invalid parameter configurations.
    """
    if feature_dimension < 2:
        raise ValueError("feature_dimension must be >= 2.")
    if reps < 1:
        raise ValueError("reps must be >= 1.")
    n = int(feature_dimension)

    # Resolve entanglement pairs
    pairs = _resolve_entanglement(n, entanglement)

    # Validate coefficient structure
    if shared_coeffs:
        coeffs_per_qubit: Sequence[Sequence[float]] = [list(single_coeffs)] * n
    else:
        if not isinstance(single_coeffs, Sequence) or isinstance(single_coeffs, str):
            raise ValueError("single_coeffs must be a sequence of sequences when shared_coeffs=False.")
        if len(single_coeffs)!= n:
            raise ValueError("Length of single_coeffs must match feature_dimension when shared_coeffs=False.")
        coeffs_per_qubit = [list(c) for c in single_coeffs]

    # Normalisation factor
    scale = 1 / sqrt(n) if normalize else 1
    scaled_pair_weight = pair_weight * (scale ** 2)

    qc = QuantumCircuit(n, name=name or "ZZFeatureMapPolyControlledModification")
    x = ParameterVector(parameter_prefix, n)

    # Optional pre‑rotation
    if apply_prepost_rotations:
        for q in range(n):
            qc.ry(pi / 2, q)

    for rep in range(reps):
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
            coeffs = coeffs_per_qubit[i]
            # Polynomial evaluation: φ1(x) = Σ c_k * (x * scale)^(k+1)
            xi = x[i] * scale
            expr: ParameterExpression = 0
            term = xi
            for c in coeffs:
                expr += c * term
                term = term * xi
            qc.p(2 * expr, i)
        if insert_barriers:
            qc.barrier()

        # ZZ interactions via CX–P–CX
        for (i, j) in pairs:
            angle = 2 * scaled_pair_weight * x[i] * x[j]
            qc.cx(i, j)
            qc.p(angle, j)
            qc.cx(i, j)
        if insert_barriers and rep!= reps - 1:
            qc.barrier()

    # Optional post‑rotation
    if apply_prepost_rotations:
        for q in range(n):
            qc.ry(pi / 2, q)

    qc.input_params = x  # type: ignore[attr-defined]
    return qc


# ---------------------------------------------------------------------------
# OO wrapper
# ---------------------------------------------------------------------------

class ZZFeatureMapPolyControlledModification(QuantumCircuit):
    """
    OO wrapper for :func:`zz_feature_map_poly_controlled_modification`.

    Parameters
    ----------
    feature_dimension : int
    reps : int, default 2
    entanglement : str | Sequence[Tuple[int, int]] | Callable
    single_coeffs : Sequence[float] | Sequence[Sequence[float]]
    pair_weight : float, default 1.0
    basis : str, default "h"
    parameter_prefix : str, default "x"
    insert_barriers : bool, default False
    normalize : bool, default False
    shared_coeffs : bool, default True
    apply_prepost_rotations : bool, default False
    name : str, default "ZZFeatureMapPolyControlledModification"
    """

    def __init__(
        self,
        feature_dimension: int,
        reps: int = 2,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
        single_coeffs: Sequence[float] | Sequence[Sequence[float]] = (1.0,),
        pair_weight: float = 1.0,
        basis: str = "h",
        parameter_prefix: str = "x",
        insert_barriers: bool = False,
        normalize: bool = False,
        shared_coeffs: bool = True,
        apply_prepost_rotations: bool = False,
        name: str = "ZZFeatureMapPolyControlledModification",
    ) -> None:
        built = zz_feature_map_poly_controlled_modification(
            feature_dimension,
            reps,
            entanglement,
            single_coeffs,
            pair_weight,
            basis,
            parameter_prefix,
            insert_barriers,
            normalize,
            shared_coeffs,
            apply_prepost_rotations,
            name=name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)
        self.input_params = built.input_params  # type: ignore[attr-defined]


__all__ = [
    "ZZFeatureMapPolyControlledModification",
    "zz_feature_map_poly_controlled_modification",
]
