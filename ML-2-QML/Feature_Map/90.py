"""
Extended ZZ‑FeatureMap with higher‑order interactions, configurable pre‑rotation depth, and optional normalisation.
Provides a functional builder `zz_feature_map_extended` and a subclass `ZZFeatureMapExtended` that are fully compatible with Qiskit data‑encoding workflows.
"""

from __future__ import annotations

import itertools
import math
from typing import Callable, Iterable, List, Sequence, Tuple, Optional

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterExpression, ParameterVector

# --------------------------------------------------------------------------- #
# Utility helpers
# --------------------------------------------------------------------------- #

def _resolve_entanglement(
    num_qubits: int,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]],
    interaction_order: int = 2,
) -> List[Tuple[int,...]]:
    """
    Translate an entanglement specification into a list of qubit tuples.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]]
        Entanglement specification.
    interaction_order : int, default=2
        Size of the tuples to generate (2 for pairwise, 3 for three‑body, etc.).

    Returns
    -------
    List[Tuple[int,...]]
        List of tuples specifying qubit indices that participate in each interaction.

    Raises
    ------
    ValueError
        If the specification is unsupported or results in invalid tuples.
    """
    if isinstance(entanglement, str):
        if entanglement == "full":
            return list(itertools.combinations(range(num_qubits), interaction_order))
        if entanglement == "linear":
            if interaction_order!= 2:
                raise ValueError("Linear entanglement is defined only for pairwise interactions.")
            return [(i, i + 1) for i in range(num_qubits - 1)]
        if entanglement == "circular":
            if interaction_order!= 2:
                raise ValueError("Circular entanglement is defined only for pairwise interactions.")
            pairs = [(i, i + 1) for i in range(num_qubits - 1)]
            if num_qubits > 2:
                pairs.append((num_qubits - 1, 0))
            return pairs
        raise ValueError(f"Unknown entanglement string: {entanglement!r}")
    if callable(entanglement):
        pairs = list(entanglement(num_qubits))
        return [(int(i),) for i in pairs]  # type: ignore[arg-type]
    # sequence of tuples
    tuples = [(int(i),) for i in entanglement]  # type: ignore[arg-type]
    # basic validation
    for tup in tuples:
        if len(tup)!= interaction_order:
            raise ValueError(f"Entanglement tuple {tup} does not match interaction order {interaction_order}.")
        if len(set(tup))!= interaction_order:
            raise ValueError(f"Entanglement tuple {tup} contains duplicate qubits.")
        if any(not (0 <= q < num_qubits) for q in tup):
            raise ValueError(f"Entanglement tuple {tup} contains out‑of‑range indices for n={num_qubits}.")
    return tuples

def _default_map_1(x: ParameterExpression) -> ParameterExpression:
    """Default single‑qubit mapping φ₁(x) = x."""
    return x

def _default_map_2(x: ParameterExpression, y: ParameterExpression) -> ParameterExpression:
    """Default pairwise mapping φ₂(x, y) = (π − x)(π − y)."""
    return (math.pi - x) * (math.pi - y)

def _default_map_n(*xs: ParameterExpression) -> ParameterExpression:
    """Default n‑body mapping φₙ(x₁,…,xₙ) = ∏ᵢ (π − xᵢ)."""
    prod = 1
    for xi in xs:
        prod *= (math.pi - xi)
    return prod

# --------------------------------------------------------------------------- #
# Functional builder
# --------------------------------------------------------------------------- #

def zz_feature_map_extended(
    feature_dimension: int,
    reps: int = 2,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
    interaction_order: int = 2,
    data_map_func: Optional[Callable[[Sequence[ParameterExpression]], ParameterExpression]] = None,
    parameter_prefix: str = "x",
    insert_barriers: bool = False,
    normalize: bool = False,
    pre_rotation_depth: int = 0,
    pre_rotation_map_func: Optional[Callable[[Sequence[ParameterExpression]], ParameterExpression]] = None,
    name: str | None = None,
) -> QuantumCircuit:
    """
    Build an extended ZZ‑feature‑map circuit.

    The circuit structure per repetition is:
        [pre‑rotation] → H → P(2·φ₁) on each qubit → n‑body entanglement via
        multi‑controlled X–P–X (or CX–P–CX for pairwise) → optional barriers.

    Parameters
    ----------
    feature_dimension : int
        Number of classical features / qubits.
    reps : int, default=2
        Number of repetitions of the feature‑map block.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]]
        Entanglement pattern. Supported strings: "full", "linear", "circular".
    interaction_order : int, default=2
        Size of the tuples defining each interaction (2 for pairwise, 3 for three‑body, etc.).
    data_map_func : Callable[[Sequence[ParameterExpression]], ParameterExpression], optional
        User‑supplied mapping from raw features to circuit parameters.
    parameter_prefix : str, default="x"
        Prefix used for the ParameterVector.
    insert_barriers : bool, default=False
        Whether to insert barriers between logical sections.
    normalize : bool, default=False
        If True, scales input features by π/2 before mapping.
    pre_rotation_depth : int, default=0
        Number of pre‑rotation layers applied before the main block.
    pre_rotation_map_func : Callable[[Sequence[ParameterExpression]], ParameterExpression], optional
        Mapping for the pre‑rotation angles (default is identity).
    name : str | None, default=None
        Optional name for the circuit.

    Returns
    -------
    QuantumCircuit
        The constructed feature‑map circuit.

    Raises
    ------
    ValueError
        If feature_dimension is insufficient for the chosen interaction_order.
    """
    if feature_dimension < interaction_order:
        raise ValueError(
            f"feature_dimension ({feature_dimension}) must be >= interaction_order ({interaction_order})."
        )

    n = int(feature_dimension)
    qc = QuantumCircuit(n, name=name or "ZZFeatureMapExtended")

    # Parameter vector for the raw data
    x = ParameterVector(parameter_prefix, n)

    # Wrap the user‑supplied mapping to include optional normalisation
    def _wrap_map(func: Callable[[Sequence[ParameterExpression]], ParameterExpression]) -> Callable:
        if normalize:
            return lambda *args: (math.pi / 2) * func(*args)
        return func

    # Define mapping functions for each interaction order
    if data_map_func is None:
        map1 = _default_map_1
        map2 = _default_map_2
        mapN = _default_map_n
    else:
        map1 = lambda xi: _wrap_map(data_map_func)([xi])
        map2 = lambda xi, xj: _wrap_map(data_map_func)([xi, xj])
        mapN = lambda *xs: _wrap_map(data_map_func)(list(xs))

    # Pre‑rotation mapping
    if pre_rotation_map_func is None:
        pre_map = _default_map_1
    else:
        pre_map = _wrap_map(pre_rotation_map_func)

    # Resolve interaction tuples
    interaction_tuples = _resolve_entanglement(n, entanglement, interaction_order)

    for rep in range(int(reps)):
        # Pre‑rotation layers
        for _ in range(pre_rotation_depth):
            for i in range(n):
                qc.rz(2 * pre_map(x[i]), i)
            if insert_barriers:
                qc.barrier()

        # Basis preparation
        qc.h(range(n))
        if insert_barriers:
            qc.barrier()

        # Single‑qubit phase rotations
        for i in range(n):
            qc.p(2 * map1(x[i]), i)
        if insert_barriers:
            qc.barrier()

        # n‑body entanglement
        for tup in interaction_tuples:
            if interaction_order == 2:
                i, j = tup
                angle = 2 * map2(x[i], x[j])
                qc.cx(i, j)
                qc.p(angle, j)
                qc.cx(i, j)
            else:
                # Use multi‑controlled X (mcx) with the last qubit as target
                target = tup[-1]
                controls = list(tup[:-1])
                angle = 2 * mapN(*[x[q] for q in tup])
                qc.mcx(controls, target, mode="noancilla")
                qc.p(angle, target)
                qc.mcx(controls, target, mode="noancilla")

        if insert_barriers and rep!= reps - 1:
            qc.barrier()

    qc.input_params = x  # type: ignore[attr-defined]
    return qc


# --------------------------------------------------------------------------- #
# Object‑oriented wrapper
# --------------------------------------------------------------------------- #

class ZZFeatureMapExtended(QuantumCircuit):
    """
    Object‑oriented wrapper for the extended ZZ‑FeatureMap.

    Parameters are identical to :func:`zz_feature_map_extended`.  The circuit
    is composed in place during construction, and the public attribute
    ``input_params`` exposes the ParameterVector for easy binding.
    """

    def __init__(
        self,
        feature_dimension: int,
        reps: int = 2,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
        interaction_order: int = 2,
        data_map_func: Optional[Callable[[Sequence[ParameterExpression]], ParameterExpression]] = None,
        parameter_prefix: str = "x",
        insert_barriers: bool = False,
        normalize: bool = False,
        pre_rotation_depth: int = 0,
        pre_rotation_map_func: Optional[Callable[[Sequence[ParameterExpression]], ParameterExpression]] = None,
        name: str = "ZZFeatureMapExtended",
    ) -> None:
        built = zz_feature_map_extended(
            feature_dimension=feature_dimension,
            reps=reps,
            entanglement=entanglement,
            interaction_order=interaction_order,
            data_map_func=data_map_func,
            parameter_prefix=parameter_prefix,
            insert_barriers=insert_barriers,
            normalize=normalize,
            pre_rotation_depth=pre_rotation_depth,
            pre_rotation_map_func=pre_rotation_map_func,
            name=name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)
        self.input_params = built.input_params  # type: ignore[attr-defined]


__all__ = ["zz_feature_map_extended", "ZZFeatureMapExtended"]
