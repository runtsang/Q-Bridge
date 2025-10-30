"""Polynomial ZZFeatureMap variant with controlled modifications: shared parameters, optional rotations, and data scaling."""
from __future__ import annotations

from math import pi
from typing import Callable, List, Sequence, Tuple, Union

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector, ParameterExpression


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _resolve_entanglement(
    num_qubits: int,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]],
) -> List[Tuple[int, int]]:
    """Return a list of two‑qubit pairs according to an entanglement spec.

    The spec can be a string ("full", "linear", "circular"),
    an explicit list of index pairs, or a callable that
    generates the pairs given the qubit count.
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
        return [(int(i), int(j)) for i, j in entanglement(num_qubits)]

    # explicit sequence
    pairs = [(int(i), int(j)) for i, j in entanglement]  # type: ignore[arg-type]
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
    coeffs: Sequence[float] = (1.0,),
    pair_weight: float = 1.0,
    shared_coeff: float = 1.0,
    feature_scale: Union[float, None] = None,
    pre_rotation_angle: Union[float, None] = None,
    post_rotation_angle: Union[float, None] = None,
    basis: str = "h",
    parameter_prefix: str = "x",
    insert_barriers: bool = False,
    name: str | None = None,
) -> QuantumCircuit:
    """Construct a controlled‑modified polynomial ZZ feature map.

    The map applies a shared scaling factor to all single‑qubit phases,
    optional pre‑ and post‑rotations, and a global scaling of the input
    data.  The entanglement pattern and polynomial coefficients are
    fully configurable.

    Parameters
    ----------
    feature_dimension : int
        Number of classical features / qubits. Must be >= 2.
    reps : int
        Number of repetitions of the feature‑map layer. Defaults to 2.
    entanglement : str | Sequence[Tuple[int, int]] | Callable
        Specification of which qubit pairs receive ZZ interactions.
    coeffs : Sequence[float]
        Polynomial coefficients for the single‑qubit phase φ₁(x) = Σ c_k x^{k+1}.
    pair_weight : float
        Weight applied to the pairwise phase φ₂(x, y) = x y.
    shared_coeff : float
        Global scaling factor applied to all single‑qubit phases.
    feature_scale : float | None
        Optional global scaling applied to every feature before mapping.
    pre_rotation_angle : float | None
        Optional RZ rotation applied to every qubit before each repetition.
    post_rotation_angle : float | None
        Optional RZ rotation applied to every qubit after each repetition.
    basis : str
        Basis preparation: "h" for Hadamard, "ry" for RY(π/2).
    parameter_prefix : str
        Prefix used for the ParameterVector names.
    insert_barriers : bool
        Whether to insert barriers between logical blocks.
    name : str | None
        Optional name for the returned QuantumCircuit.

    Returns
    -------
    QuantumCircuit
        The constructed feature‑map circuit.  The circuit has an ``input_params``
        attribute containing the ParameterVector for user binding.
    """
    if feature_dimension < 2:
        raise ValueError("feature_dimension must be >= 2.")
    n = int(feature_dimension)
    qc = QuantumCircuit(n, name=name or "ZZFeatureMapPolyControlled")

    # Parameter vector for classical data
    params = ParameterVector(parameter_prefix, n)

    # Optional feature scaling
    def scale_expr(val: ParameterExpression) -> ParameterExpression:
        if feature_scale is None:
            return val
        return val * feature_scale

    # Polynomial map for single‑qubit phase
    def map1(xi: ParameterExpression) -> ParameterExpression:
        expr: ParameterExpression = 0
        power: ParameterExpression = scale_expr(xi)
        for c in coeffs:
            expr += c * power
            power = power * scale_expr(xi)
        return shared_coeff * expr

    # Pairwise phase map
    def map2(xi: ParameterExpression, xj: ParameterExpression) -> ParameterExpression:
        return pair_weight * scale_expr(xi) * scale_expr(xj)

    pairs = _resolve_entanglement(n, entanglement)

    for rep in range(int(reps)):
        # Optional pre‑rotation
        if pre_rotation_angle is not None:
            for q in range(n):
                qc.rz(pre_rotation_angle, q)

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
            qc.p(2 * map1(params[i]), i)

        if insert_barriers:
            qc.barrier()

        # ZZ interactions via CX‑P‑CX
        for (i, j) in pairs:
            angle = 2 * map2(params[i], params[j])
            qc.cx(i, j)
            qc.p(angle, j)
            qc.cx(i, j)

        # Optional post‑rotation
        if post_rotation_angle is not None:
            for q in range(n):
                qc.rz(post_rotation_angle, q)

        if insert_barriers and rep!= reps - 1:
            qc.barrier()

    # Attach the ParameterVector for easy binding
    qc.input_params = params  # type: ignore[attr-defined]
    return qc


class ZZFeatureMapPolyControlled(QuantumCircuit):
    """Object‑oriented wrapper for the controlled‑modified polynomial ZZ feature map.

    The class inherits from :class:`qiskit.QuantumCircuit` and builds the circuit
    during initialization.  It exposes the same parameters as the functional
    helper and populates an ``input_params`` attribute for convenient
    ``bind_parameters`` calls.
    """

    def __init__(
        self,
        feature_dimension: int,
        reps: int = 2,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
        coeffs: Sequence[float] = (1.0,),
        pair_weight: float = 1.0,
        shared_coeff: float = 1.0,
        feature_scale: Union[float, None] = None,
        pre_rotation_angle: Union[float, None] = None,
        post_rotation_angle: Union[float, None] = None,
        basis: str = "h",
        parameter_prefix: str = "x",
        insert_barriers: bool = False,
        name: str = "ZZFeatureMapPolyControlled",
    ) -> None:
        built = zz_feature_map_poly_controlled(
            feature_dimension,
            reps,
            entanglement,
            coeffs,
            pair_weight,
            shared_coeff,
            feature_scale,
            pre_rotation_angle,
            post_rotation_angle,
            basis,
            parameter_prefix,
            insert_barriers,
            name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)
        self.input_params = built.input_params  # type: ignore[attr-defined]


__all__ = ["ZZFeatureMapPolyControlled", "zz_feature_map_poly_controlled"]
