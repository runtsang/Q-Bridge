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
    entanglement: Union[str, Sequence[Tuple[int, int]], Callable[[int], Sequence[Tuple[int, int]]]],
) -> List[Tuple[int, int]]:
    """
    Resolve an entanglement specification into a list of two‑qubit pairs.

    Supported specifications:

    * ``"full"``          – all‑to‑all pairs (i < j)
    * ``"linear"``        – nearest neighbours (0,1), (1,2), …
    * ``"circular"``      – linear + wrap‑around (n‑1,0) when n > 2
    * explicit list ``[(0, 1), (2, 3)]``
    * callable ``f(n)`` → sequence of pairs

    Raises
    ------
    ValueError
        If an unsupported string or out‑of‑range pair is supplied.
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
# Feature‑map definition
# ---------------------------------------------------------------------------

def zz_feature_map_poly_extended(
    feature_dimension: int,
    reps: int = 2,
    entanglement: Union[str, Sequence[Tuple[int, int]], Callable[[int], Sequence[Tuple[int, int]]]] = "full",
    single_coeffs: Sequence[float] = (1.0,),
    pair_weight: float = 1.0,
    pair_exponent: int = 1,
    basis: str = "h",
    parameter_prefix: str = "x",
    insert_barriers: bool = False,
    name: str | None = None,
    data_scaling: float = 1.0,
    pre_rotation: str | None = None,
    post_rotation: str | None = None,
) -> QuantumCircuit:
    """
    Extended polynomial ZZ feature map.

    Parameters
    ----------
    feature_dimension : int
        Number of classical features / qubits. Must be ≥ 2.
    reps : int, default=2
        Number of repetitions of the entangling block.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]]
        Entanglement pattern. See :func:`_resolve_entanglement` for details.
    single_coeffs : Sequence[float], default=(1.0,)
        Coefficients for the polynomial mapping φ₁(x) = Σ c_k · (x)^{k+1}.
    pair_weight : float, default=1.0
        Overall weight for pairwise interactions.
    pair_exponent : int, default=1
        Exponent applied to the product term: φ₂(x, y) = pair_weight · (x·y)^{pair_exponent}.
    basis : str, default="h"
        Basis preparation: ``"h"`` applies Hadamards, ``"ry"`` applies RY(π/2).
    parameter_prefix : str, default="x"
        Prefix used for the parameter vector.
    insert_barriers : bool, default=False
        Whether to insert barriers between logical sections for readability.
    name : str | None, default=None
        Optional circuit name.
    data_scaling : float, default=1.0
        Scale factor applied to all input features before mapping.
    pre_rotation : str | None, default=None
        Optional pre‑rotation applied to all qubits (``"ry"``, ``"rx"``, ``"rz"``, ``"h"``, or ``None``).
    post_rotation : str | None, default=None
        Optional post‑rotation applied to all qubits after entanglement.

    Returns
    -------
    QuantumCircuit
        Parameterised circuit ready for binding with a feature vector.

    Notes
    -----
    * The circuit remains fully compatible with Qiskit’s data‑encoding utilities:
      the resulting `QuantumCircuit` exposes the attribute ``input_params`` which can be passed to
      :meth:`QuantumCircuit.bind_parameters_dict`.
    * Higher‑order pairwise interactions are achieved via ``pair_exponent``.
    * Optional pre‑ and post‑rotations allow further control over the feature embedding.
    """
    # Input validation
    if feature_dimension < 2:
        raise ValueError("feature_dimension must be >= 2.")
    if reps < 1:
        raise ValueError("reps must be >= 1.")
    if pair_exponent < 1:
        raise ValueError("pair_exponent must be >= 1.")
    if data_scaling <= 0:
        raise ValueError("data_scaling must be positive.")
    allowed_rotations = {None, "ry", "rx", "rz", "h"}
    if pre_rotation not in allowed_rotations:
        raise ValueError(f"Unsupported pre_rotation: {pre_rotation!r}.")
    if post_rotation not in allowed_rotations:
        raise ValueError(f"Unsupported post_rotation: {post_rotation!r}.")

    n = int(feature_dimension)
    qc = QuantumCircuit(n, name=name or "ZZFeatureMapPolyExtended")
    x = ParameterVector(parameter_prefix, n)

    # Helper to apply a rotation gate to all qubits
    def _apply_rotation(gate: str | None, angle: float | None = None) -> None:
        if gate is None:
            return
        if gate == "ry":
            angle = angle if angle is not None else pi / 2
            for q in range(n):
                qc.ry(angle, q)
        elif gate == "rx":
            angle = angle if angle is not None else pi / 2
            for q in range(n):
                qc.rx(angle, q)
        elif gate == "rz":
            angle = angle if angle is not None else pi / 2
            for q in range(n):
                qc.rz(angle, q)
        elif gate == "h":
            qc.h(range(n))
        else:
            raise ValueError(f"Unsupported rotation gate: {gate!r}")

    pairs = _resolve_entanglement(n, entanglement)

    for rep in range(int(reps)):
        # Optional pre‑rotation
        _apply_rotation(pre_rotation)

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

        # Single‑qubit phase gates
        for i in range(n):
            # φ₁(xᵢ) = Σ c_k · (data_scaling·xᵢ)^{k+1}
            expr: ParameterExpression = 0
            power_term = data_scaling * x[i]
            for c in single_coeffs:
                expr += c * power_term
                power_term *= data_scaling * x[i]
            qc.p(2 * expr, i)

        if insert_barriers:
            qc.barrier()

        # ZZ‑interaction block
        for (i, j) in pairs:
            # φ₂(xᵢ, xⱼ) = pair_weight · (data_scaling·xᵢ·xⱼ)^{pair_exponent}
            angle = 2 * pair_weight * (data_scaling * x[i] * x[j]) ** pair_exponent
            qc.cx(i, j)
            qc.p(angle, j)
            qc.cx(i, j)

        if insert_barriers and rep!= reps - 1:
            qc.barrier()

        # Optional post‑rotation
        _apply_rotation(post_rotation)

    qc.input_params = x  # type: ignore[attr-defined]
    return qc


class ZZFeatureMapPolyExtended(QuantumCircuit):
    """
    OO wrapper for :func:`zz_feature_map_poly_extended`.

    Parameters are identical to the functional API.
    """
    def __init__(
        self,
        feature_dimension: int,
        reps: int = 2,
        entanglement: Union[str, Sequence[Tuple[int, int]], Callable[[int], Sequence[Tuple[int, int]]]] = "full",
        single_coeffs: Sequence[float] = (1.0,),
        pair_weight: float = 1.0,
        pair_exponent: int = 1,
        basis: str = "h",
        parameter_prefix: str = "x",
        insert_barriers: bool = False,
        name: str | None = None,
        data_scaling: float = 1.0,
        pre_rotation: str | None = None,
        post_rotation: str | None = None,
    ) -> None:
        built = zz_feature_map_poly_extended(
            feature_dimension,
            reps,
            entanglement,
            single_coeffs,
            pair_weight,
            pair_exponent,
            basis,
            parameter_prefix,
            insert_barriers,
            name,
            data_scaling,
            pre_rotation,
            post_rotation,
        )
        super().__init__(built.num_qubits, name=name or "ZZFeatureMapPolyExtended")
        self.compose(built, inplace=True)
        self.input_params = built.input_params  # type: ignore[attr-defined]


__all__ = ["ZZFeatureMapPolyExtended", "zz_feature_map_poly_extended"]
