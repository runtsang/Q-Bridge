from __future__ import annotations

from math import pi
from typing import Callable, Iterable, List, Sequence, Tuple, Union

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterExpression, ParameterVector


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _resolve_entanglement(
    num_qubits: int,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]],
) -> List[Tuple[int, int]]:
    """
    Resolve an entanglement specification into a list of two‑qubit pairs.

    Supported specs:
      * ``"full"``   – all‑to‑all pairs (i < j)
      * ``"linear"`` – nearest neighbours (0,1), (1,2), …
      * ``"circular"`` – linear plus wrap‑around (n‑1,0) if n > 2
      * explicit list of pairs ``[(0, 2), (1, 3)]``
      * callable ``f(num_qubits) -> sequence of (i, j)``

    Raises
    ------
    ValueError
        If the specification is invalid or contains out‑of-range indices.
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

    pairs = [(int(i), int(j)) for (i, j) in entanglement]  # type: ignore[arg-type]
    for (i, j) in pairs:
        if i == j:
            raise ValueError("Entanglement pairs must connect distinct qubits.")
        if not (0 <= i < num_qubits and 0 <= j < num_qubits):
            raise ValueError(f"Entanglement pair {(i, j)} out of range for n={num_qubits}.")
    return pairs


def _validate_shared_parameter(
    param: Sequence[float] | float,
    expected_length: int,
    name: str,
    shared: bool,
) -> List[float]:
    """
    Ensure a parameter is either a scalar or a sequence of the expected length.
    If a scalar is supplied but ``shared`` is False, it is broadcast to the length.
    """
    if shared:
        if isinstance(param, (float, int)):
            return [float(param)] * expected_length
        if len(param)!= 1:
            raise ValueError(f"{name} must be a single value for shared mode.")
        return [float(param[0])] * expected_length
    else:
        if isinstance(param, (float, int)):
            return [float(param)] * expected_length
        if len(param)!= expected_length:
            raise ValueError(f"{name} length {len(param)} does not match expected {expected_length}.")
        return [float(v) for v in param]


# ---------------------------------------------------------------------------
# Feature‑map construction
# ---------------------------------------------------------------------------

def zz_feature_map_poly_controlled(
    feature_dimension: int,
    *,
    reps: int = 2,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
    single_coeffs: Sequence[float] = (1.0,),
    pair_weight: Union[float, Sequence[float]] = 1.0,
    basis: str = "h",
    parameter_prefix: str = "x",
    insert_barriers: bool = False,
    name: str | None = None,
    data_scaling: float | None = None,
    normalize: bool = False,
    shared_single_coeffs: bool = False,
    shared_pair_weight: bool = False,
) -> QuantumCircuit:
    """
    Controlled‑modification polynomial ZZ feature map.

    Parameters
    ----------
    feature_dimension : int
        Number of qubits / dimensionality of the classical feature vector.
    reps : int, default 2
        Number of repetitions of the basis / phase / entanglement block.
    entanglement : str | sequence | callable, default "full"
        Specification of two‑qubit coupling pairs.
    single_coeffs : sequence, default (1.0,)
        Polynomial coefficients for the single‑qubit phase φ₁(x).
        If ``shared_single_coeffs`` is True, a single coefficient is applied to all qubits.
    pair_weight : float | sequence, default 1.0
        Weight applied to the pairwise product φ₂(xᵢ, xⱼ).
        If ``shared_pair_weight`` is True, a single weight is used for all pairs.
    basis : str, default "h"
        Basis preparation: ``"h"`` for Hadamard, ``"ry"`` for RY(π/2).
    parameter_prefix : str, default "x"
        Prefix for the parameter vector names.
    insert_barriers : bool, default False
        Insert barriers between logical blocks for readability.
    name : str | None, default None
        Optional name for the resulting QuantumCircuit.
    data_scaling : float | None, default None
        Scaling factor applied to each classical feature before mapping.
    normalize : bool, default False
        If True, normalise features to the interval [0, 1] using ``data_scaling`` as the
        maximum absolute value. Raises if ``data_scaling`` is None.
    shared_single_coeffs : bool, default False
        If True, ``single_coeffs`` must contain a single value shared across all qubits.
    shared_pair_weight : bool, default False
        If True, ``pair_weight`` must be a single value shared across all pairs.

    Returns
    -------
    QuantumCircuit
        A parameterised feature‑map circuit compatible with Qiskit data encoding.

    Notes
    -----
    * The single‑qubit phase is defined as
      φ₁(x) = Σ_k c_k · x^{k+1}
      where the coefficients `c_k` are taken from ``single_coeffs``.
    * The pairwise phase is
      φ₂(xᵢ, xⱼ) = w · xᵢ · xⱼ
      with weight `w` from ``pair_weight``.
    * The circuit replaces the CX‑P‑CX pattern with a single CP gate for each pair.
    * Data scaling and normalisation apply to all classical features uniformly.
    """
    if feature_dimension < 2:
        raise ValueError("feature_dimension must be >= 2.")
    n = int(feature_dimension)

    if reps < 1:
        raise ValueError("reps must be at least 1.")

    pairs = _resolve_entanglement(n, entanglement)

    # Validate and broadcast single‑qubit coefficients
    single_coeffs_list = _validate_shared_parameter(
        single_coeffs, n, "single_coeffs", shared_single_coeffs
    )

    # Validate and broadcast pair weights
    pair_weights_list = _validate_shared_parameter(
        pair_weight, len(pairs), "pair_weight", shared_pair_weight
    )

    # Parameter vector for single‑qubit data
    if shared_single_coeffs:
        x = ParameterVector(parameter_prefix, 1)
    else:
        x = ParameterVector(parameter_prefix, n)

    # Data scaling / normalisation
    scale_factor = 1.0
    if data_scaling is not None:
        if data_scaling <= 0:
            raise ValueError("data_scaling must be positive.")
        scale_factor = 1.0 / data_scaling if normalize else data_scaling
    elif normalize:
        raise ValueError("normalize=True requires a positive data_scaling value.")

    # Helper functions for the polynomial maps
    def map1(xi: ParameterExpression) -> ParameterExpression:
        """Polynomial φ₁(x) = Σ_k c_k · x^{k+1}."""
        expr: ParameterExpression = 0
        p = xi
        for c in single_coeffs_list:
            expr += c * p
            p = p * xi
        return expr

    def map2(xi: ParameterExpression, xj: ParameterExpression, weight: float) -> ParameterExpression:
        """Pairwise product φ₂(xᵢ, xⱼ) = w · xᵢ · xⱼ."""
        return weight * xi * xj

    qc = QuantumCircuit(n, name=name or "ZZFeatureMapPolyControlled")

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
            xi = x[0] if shared_single_coeffs else x[i]
            angle = 2 * map1(xi * scale_factor)
            qc.p(angle, i)

        if insert_barriers:
            qc.barrier()

        # Pairwise entanglement via controlled‑phase
        for (idx, (i, j)) in enumerate(pairs):
            weight = pair_weights_list[idx]
            angle = 2 * map2(x[i] * scale_factor, x[j] * scale_factor, weight)
            qc.cp(angle, i, j)

        if insert_barriers and rep!= reps - 1:
            qc.barrier()

    qc.input_params = x  # type: ignore[attr-defined]
    return qc


# ---------------------------------------------------------------------------
# Class‑style wrapper
# ---------------------------------------------------------------------------

class ZZFeatureMapPolyControlled(QuantumCircuit):
    """
    OO wrapper for the controlled‑modification polynomial ZZ feature map.
    Parameters are forwarded to :func:`zz_feature_map_poly_controlled`.
    """

    def __init__(
        self,
        feature_dimension: int,
        *,
        reps: int = 2,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
        single_coeffs: Sequence[float] = (1.0,),
        pair_weight: Union[float, Sequence[float]] = 1.0,
        basis: str = "h",
        parameter_prefix: str = "x",
        insert_barriers: bool = False,
        name: str = "ZZFeatureMapPolyControlled",
        data_scaling: float | None = None,
        normalize: bool = False,
        shared_single_coeffs: bool = False,
        shared_pair_weight: bool = False,
    ) -> None:
        built = zz_feature_map_poly_controlled(
            feature_dimension,
            reps=reps,
            entanglement=entanglement,
            single_coeffs=single_coeffs,
            pair_weight=pair_weight,
            basis=basis,
            parameter_prefix=parameter_prefix,
            insert_barriers=insert_barriers,
            name=name,
            data_scaling=data_scaling,
            normalize=normalize,
            shared_single_coeffs=shared_single_coeffs,
            shared_pair_weight=shared_pair_weight,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)
        self.input_params = built.input_params  # type: ignore[attr-defined]


__all__ = ["ZZFeatureMapPolyControlled", "zz_feature_map_poly_controlled"]
