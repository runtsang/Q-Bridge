"""Shared-parameter polynomial ZZ feature map with symmetric entanglement.

This module implements a controlled‑modification scaling of the original
`ZZFeatureMapPoly`.  The key differences are:

* **Shared coefficients** – a single `single_coeff` and a single
  `pair_weight` control all single‑qubit and two‑qubit phases.
* **Global scaling** – an optional `global_scale` multiplies every feature
  before mapping, allowing the user to normalise or otherwise rescale data
  without changing the circuit topology.
* **Symmetric entanglement** – the default "full" entanglement is retained,
  but the implementation explicitly ensures each pair is applied exactly
  once, preserving symmetry even for custom pair lists.
* **Pre‑ and post‑basis options** – users may choose Hadamard or RY(π/2)
  preparation, and can insert optional barriers for debugging.

The module exposes a helper function `zz_feature_map_poly_shared` and a
`QuantumCircuit` subclass `ZZFeatureMapPolyShared`.  Both accept a
`ParameterVector` of length `feature_dimension` and bindable parameters
for the shared coefficients and global scale.

The design follows Qiskit’s conventions and includes detailed error
messages for invalid arguments.  The circuit supports parameter binding
with any classical feature vector of appropriate length.

Typical usage:

```python
from zz_feature_map_poly_shared import zz_feature_map_poly_shared

# Build a 4‑qubit circuit with shared coefficients
qc = zz_feature_map_poly_shared(
    feature_dimension=4,
    reps=3,
    single_coeff=0.5,
    pair_weight=1.2,
    global_scale=1.0,
    basis="h",
    entanglement="full",
    insert_barriers=False,
)

# Bind classical data
params = dict(zip(qc.input_params, [0.1, 0.2, 0.3, 0.4]))
qc.bind_parameters(params)
```

"""

from __future__ import annotations

from math import pi
from typing import Callable, Iterable, List, Sequence, Tuple, Union

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterExpression, ParameterVector


# ---------------------------------------------------------------------------
# Utility: Entanglement pair resolution
# ---------------------------------------------------------------------------

def _resolve_entanglement(
    num_qubits: int,
    entanglement: Union[
        str,
        Sequence[Tuple[int, int]],
        Callable[[int], Iterable[Tuple[int, int]]],
    ],
) -> List[Tuple[int, int]]:
    """Return a list of two‑qubit pairs according to an entanglement spec.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Iterable[Tuple[int, int]]]
        Specification of which qubit pairs should be entangled.

    Returns
    -------
    List[Tuple[int, int]]
        Validated list of unique, distinct qubit pairs.

    Raises
    ------
    ValueError
        If the spec is invalid or contains out‑of‑range indices.
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
    else:
        pairs = list(entanglement)

    validated: List[Tuple[int, int]] = []
    for (i, j) in pairs:
        if i == j:
            raise ValueError(f"Entanglement pair {(i, j)} connects a qubit to itself.")
        if not (0 <= i < num_qubits and 0 <= j < num_qubits):
            raise ValueError(f"Entanglement pair {(i, j)} out of range for n={num_qubits}.")
        validated.append((int(i), int(j)))
    # Remove duplicates while preserving order
    seen = set()
    unique: List[Tuple[int, int]] = []
    for pair in validated:
        if pair not in seen:
            seen.add(pair)
            unique.append(pair)
    return unique


# ---------------------------------------------------------------------------
# Core helper function
# ---------------------------------------------------------------------------

def zz_feature_map_poly_shared(
    feature_dimension: int,
    reps: int = 2,
    entanglement: Union[
        str,
        Sequence[Tuple[int, int]],
        Callable[[int], Iterable[Tuple[int, int]]],
    ] = "full",
    single_coeff: float = 1.0,
    pair_weight: float = 1.0,
    basis: str = "h",
    parameter_prefix: str = "x",
    global_scale: float = 1.0,
    insert_barriers: bool = False,
    name: str | None = None,
) -> QuantumCircuit:
    """Construct a shared‑parameter polynomial ZZ feature map.

    Parameters
    ----------
    feature_dimension : int
        Number of classical features and qubits.
    reps : int, default 2
        Number of feature‑map repetitions.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Iterable[Tuple[int, int]]], default "full"
        How qubit pairs are entangled.
    single_coeff : float, default 1.0
        Coefficient shared across all single‑qubit phase terms.
    pair_weight : float, default 1.0
        Weight shared across all two‑qubit ZZ phase terms.
    basis : str, default "h"
        Basis preparation: "h" for Hadamard, "ry" for RY(π/2).
    parameter_prefix : str, default "x"
        Prefix for the `ParameterVector` names.
    global_scale : float, default 1.0
        Global multiplicative factor applied to every feature before mapping.
    insert_barriers : bool, default False
        Whether to insert barriers between logical blocks.
    name : str | None, default None
        Optional circuit name.

    Returns
    -------
    QuantumCircuit
        Parameterised feature‑map circuit.

    Raises
    ------
    ValueError
        For invalid argument values.
    """
    # ---- Validation ----------------------------------------------------
    if feature_dimension < 2:
        raise ValueError("feature_dimension must be >= 2.")
    if reps < 1:
        raise ValueError("reps must be >= 1.")
    if not isinstance(single_coeff, (int, float)):
        raise TypeError("single_coeff must be a numeric type.")
    if not isinstance(pair_weight, (int, float)):
        raise TypeError("pair_weight must be a numeric type.")
    if not isinstance(global_scale, (int, float)):
        raise TypeError("global_scale must be a numeric type.")
    if basis not in {"h", "ry"}:
        raise ValueError("basis must be either 'h' or 'ry'.")

    # ---- Circuit construction -----------------------------------------
    n = int(feature_dimension)
    qc = QuantumCircuit(n, name=name or "ZZFeatureMapPolyShared")
    x = ParameterVector(parameter_prefix, n)

    # Pre‑compute entanglement pairs
    pairs = _resolve_entanglement(n, entanglement)

    # Helper lambdas for phase functions
    def single_phase(xi: ParameterExpression) -> ParameterExpression:
        return 2 * single_coeff * global_scale * xi

    def pair_phase(xi: ParameterExpression, xj: ParameterExpression) -> ParameterExpression:
        return 2 * pair_weight * (global_scale * xi) * (global_scale * xj)

    # Build each repetition
    for rep in range(reps):
        # Basis preparation
        if basis == "h":
            qc.h(range(n))
        else:
            for q in range(n):
                qc.ry(pi / 2, q)
        if insert_barriers:
            qc.barrier()

        # Single‑qubit phases
        for i in range(n):
            qc.p(single_phase(x[i]), i)
        if insert_barriers:
            qc.barrier()

        # ZZ interactions via CX–P–CX
        for (i, j) in pairs:
            qc.cx(i, j)
            qc.p(pair_phase(x[i], x[j]), j)
            qc.cx(i, j)
        if insert_barriers and rep!= reps - 1:
            qc.barrier()

    # Attach parameter metadata
    qc.input_params = x  # type: ignore[attr-defined]
    qc.single_coeff = single_coeff
    qc.pair_weight = pair_weight
    qc.global_scale = global_scale
    return qc


# ---------------------------------------------------------------------------
# OO wrapper
# ---------------------------------------------------------------------------

class ZZFeatureMapPolyShared(QuantumCircuit):
    """QuantumCircuit subclass of the shared‑parameter polynomial ZZ feature map.

    The constructor forwards all arguments to :func:`zz_feature_map_poly_shared`
    and composes the resulting circuit into the subclass instance.  The
    resulting object retains a ``input_params`` attribute for parameter
    binding and exposes ``single_coeff``, ``pair_weight`` and ``global_scale``
    for introspection.

    Parameters
    ----------
    feature_dimension : int
        Number of classical features and qubits.
    reps : int, default 2
        Number of feature‑map repetitions.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Iterable[Tuple[int, int]]], default "full"
        How qubit pairs are entangled.
    single_coeff : float, default 1.0
        Coefficient shared across all single‑qubit phase terms.
    pair_weight : float, default 1.0
        Weight shared across all two‑qubit ZZ phase terms.
    basis : str, default "h"
        Basis preparation: "h" for Hadamard, "ry" for RY(π/2).
    parameter_prefix : str, default "x"
        Prefix for the `ParameterVector` names.
    global_scale : float, default 1.0
        Global multiplicative factor applied to every feature before mapping.
    insert_barriers : bool, default False
        Whether to insert barriers between logical blocks.
    name : str, default "ZZFeatureMapPolyShared"
        Circuit name.

    Raises
    ------
    ValueError
        For invalid argument values.
    """
    def __init__(
        self,
        feature_dimension: int,
        reps: int = 2,
        entanglement: Union[
            str,
            Sequence[Tuple[int, int]],
            Callable[[int], Iterable[Tuple[int, int]]],
        ] = "full",
        single_coeff: float = 1.0,
        pair_weight: float = 1.0,
        basis: str = "h",
        parameter_prefix: str = "x",
        global_scale: float = 1.0,
        insert_barriers: bool = False,
        name: str = "ZZFeatureMapPolyShared",
    ) -> None:
        built = zz_feature_map_poly_shared(
            feature_dimension,
            reps,
            entanglement,
            single_coeff,
            pair_weight,
            basis,
            parameter_prefix,
            global_scale,
            insert_barriers,
            name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)
        self.input_params = built.input_params  # type: ignore[attr-defined]
        self.single_coeff = built.single_coeff
        self.pair_weight = built.pair_weight
        self.global_scale = built.global_scale


__all__ = ["ZZFeatureMapPolyShared", "zz_feature_map_poly_shared"]
