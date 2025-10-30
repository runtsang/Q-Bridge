"""Symmetrised polynomial ZZ feature map with shared parameters.

The circuit is a controlled modification of the original ZZFeatureMapPoly.
It introduces a single weight for all single‑qubit phases and a shared weight
for pair interactions.  The pair interaction can be chosen between a product
(x_i * x_j) or an average ((x_i + x_j)/2).  This reduces the number of
trainable parameters while preserving the expressive power of the original
feature map.  The module remains fully compatible with Qiskit data encoding
workflows: the returned circuit exposes its input parameters via the
`input_params` attribute and can be bound with a list or array of classical
features.

Supported datasets
------------------
Any dataset whose feature vectors can be mapped to real numbers
in an arbitrary interval.  The circuit does not impose any normalisation
on the input values; users are encouraged to normalise outside the
feature map if required.

Parameter constraints
---------------------
* `feature_dimension` must be >= 1.
* `single_weight` and `pair_weight` must be real numbers.
* `pair_mode` must be either ``"product"`` or ``"average"``.
* `entanglement` follows the same specifications as the original
  feature map: ``"full"``, ``"linear"``, ``"circular"``, a list of pairs,
  or a callable returning a list of pairs.
"""

from __future__ import annotations

import math
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
    """Return a list of two-qubit pairs according to a simple entanglement spec.

    Supported specs:
      - "full": all-to-all pairs (i < j)
      - "linear": nearest neighbours (0,1), (1,2),...
      - "circular": linear plus wrap‑around (n-1,0) if n > 2
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
    single_weight: float = 1.0,
    pair_weight: float = 1.0,
    pair_mode: str = "product",
    basis: str = "h",
    parameter_prefix: str = "x",
    insert_barriers: bool = False,
    name: str | None = None,
) -> QuantumCircuit:
    """Symmetrised polynomial ZZ feature map with shared parameters.

    Parameters
    ----------
    feature_dimension : int
        Number of classical features / qubits. Must be >= 1.
    reps : int, default 2
        Number of repetitions of the encoding block.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]]
        Entanglement specification.  See ``_resolve_entanglement``.
    single_weight : float, default 1.0
        Coefficient applied to every single‑qubit phase term.
    pair_weight : float, default 1.0
        Coefficient applied to every pair‑interaction term.
    pair_mode : str, default "product"
        Choice of pair interaction:
        * "product" : φ₂(xᵢ, xⱼ) = pair_weight * xᵢ * xⱼ
        * "average" : φ₂(xᵢ, xⱼ) = pair_weight * (xᵢ + xⱼ) / 2
    basis : str, default "h"
        Basis preparation before each repetition.
        * "h"   → Hadamard on all qubits
        * "ry"  → RY(π/2) on all qubits
    parameter_prefix : str, default "x"
        Prefix for the parameter vector names.
    insert_barriers : bool, default False
        If True, insert barriers between logical blocks for clarity.
    name : str | None, default None
        Optional circuit name.

    Returns
    -------
    QuantumCircuit
        Prepared feature‑map circuit with ``input_params`` attribute.
    """
    if feature_dimension < 1:
        raise ValueError("feature_dimension must be >= 1.")
    n = int(feature_dimension)
    qc = QuantumCircuit(n, name=name or "ZZFeatureMapPolySym")

    # Parameter vector for all qubits
    x = ParameterVector(parameter_prefix, n)

    # Single‑qubit phase mapping
    def map1(xi: ParameterExpression) -> ParameterExpression:
        return single_weight * xi

    # Pair‑interaction mapping
    if pair_mode == "product":
        def map2(xi: ParameterExpression, xj: ParameterExpression) -> ParameterExpression:
            return pair_weight * xi * xj
    elif pair_mode == "average":
        def map2(xi: ParameterExpression, xj: ParameterExpression) -> ParameterExpression:
            return pair_weight * (xi + xj) / 2
    else:
        raise ValueError("pair_mode must be either 'product' or 'average'.")

    # Resolve entanglement pairs
    pairs = _resolve_entanglement(n, entanglement)

    for rep in range(int(reps)):
        # Basis preparation
        if basis == "h":
            qc.h(range(n))
        elif basis == "ry":
            for q in range(n):
                qc.ry(math.pi / 2, q)
        else:
            raise ValueError("basis must be 'h' or 'ry'.")
        if insert_barriers:
            qc.barrier()

        # Single‑qubit phases
        for i in range(n):
            qc.p(2 * map1(x[i]), i)
        if insert_barriers:
            qc.barrier()

        # ZZ interactions via CX–P–CX
        for (i, j) in pairs:
            angle = 2 * map2(x[i], x[j])
            qc.cx(i, j)
            qc.p(angle, j)
            qc.cx(i, j)
        if insert_barriers and rep!= reps - 1:
            qc.barrier()

    qc.input_params = x  # type: ignore[attr-defined]
    return qc


class ZZFeatureMapPolySym(QuantumCircuit):
    """Class‑style wrapper for the symmetrised polynomial ZZ feature map.

    The constructor forwards all arguments to ``zz_feature_map_poly_sym`` and
    composes the resulting circuit into ``self``.  The ``input_params`` attribute
    of the composed circuit is preserved for parameter binding.
    """
    def __init__(
        self,
        feature_dimension: int,
        reps: int = 2,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
        single_weight: float = 1.0,
        pair_weight: float = 1.0,
        pair_mode: str = "product",
        basis: str = "h",
        parameter_prefix: str = "x",
        insert_barriers: bool = False,
        name: str = "ZZFeatureMapPolySym",
    ) -> None:
        built = zz_feature_map_poly_sym(
            feature_dimension,
            reps,
            entanglement,
            single_weight,
            pair_weight,
            pair_mode,
            basis,
            parameter_prefix,
            insert_barriers,
            name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)
        self.input_params = built.input_params  # type: ignore[attr-defined]


__all__ = ["ZZFeatureMapPolySym", "zz_feature_map_poly_sym"]
