"""ZZFeatureMapPolyControlledModification: Controlled-modification variant of the polynomial ZZ feature map.

This module provides a Qiskit-compatible quantum circuit that encodes classical data
into a parametric quantum state.  It builds upon the original polynomial ZZ
feature map but adds a number of controlled‑modification features:

* **Shared polynomial coefficients** – the single‑qubit phase map (`φ₁`) uses a
  single coefficient tuple applied to all qubits.
* **Symmetric pair interaction** – the two‑qubit phase map (`φ₂`) contains a
  symmetric polynomial in both qubits (`x·y + x²·y²`).
* **Feature scaling** – a multiplicative scaling factor that allows the user to
  pre‑scale the input features before encoding.
* **Custom mapping functions** – optional callables to replace the default
  polynomial maps.
* **Optional barrier insertion** – helps visualising the circuit structure.
* **Basis flexibility** – Hadamard or RY(π/2) preparation is supported.

The API mirrors the seed module: a functional helper (`zz_feature_map_poly_controlled_modification`)
and a class wrapper (`ZZFeatureMapPolyControlledModification`) that inherits
`QuantumCircuit`.  Both expose the `input_params` attribute for parameter
binding.

---

**Supported datasets**

The feature map is agnostic to the data distribution.  It is suitable for
datasets where the input features are real‑valued and can be normalised
to a reasonable range (e.g. [0, π] or [−π, π]).  The optional
`feature_scaling` parameter can be used to adjust the effective range
without touching the raw data.

**Parameter constraints**

* `feature_dimension` – positive integer ≥ 2.
* `reps` – positive integer.
* `entanglement` – string `"full"`, `"linear"`, `"circular"` or a user‑supplied
  list of (i, j) tuples.
* `single_coeffs` – sequence of real numbers (used identically on all qubits).
* `pair_weight` – real number; controls the strength of the two‑qubit interaction.
* `basis` – `"h"` or `"ry"`.
* `feature_scaling` – non‑negative real; defaults to 1.0.
* `custom_map1` / `custom_map2` – callables that accept `ParameterExpression`
  arguments and return a `ParameterExpression`.  If omitted, the default
  polynomial maps are used.

"""

from __future__ import annotations

from math import pi
from typing import Callable, List, Sequence, Tuple, Union, Optional

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector, ParameterExpression


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _resolve_entanglement(
    num_qubits: int,
    entanglement: Union[
        str,
        Sequence[Tuple[int, int]],
        Callable[[int], Sequence[Tuple[int, int]]],
    ],
) -> List[Tuple[int, int]]:
    """Return a list of two‑qubit pairs according to a simple entanglement spec.

    Supported specs:
      * ``"full"`` – all‑to‑all pairs (i < j)
      * ``"linear"`` – nearest neighbours (0,1),(1,2),…
      * ``"circular"`` – linear plus wrap‑around (n‑1,0) if n > 2
      * explicit list of pairs like ``[(0, 2), (1, 3)]``
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

def zz_feature_map_poly_controlled_modification(
    feature_dimension: int,
    reps: int = 2,
    entanglement: Union[
        str,
        Sequence[Tuple[int, int]],
        Callable[[int], Sequence[Tuple[int, int]]],
    ] = "full",
    single_coeffs: Sequence[float] = (1.0,),
    pair_weight: float = 1.0,
    basis: str = "h",  # "h" or "ry"
    parameter_prefix: str = "x",
    insert_barriers: bool = False,
    name: str | None = None,
    feature_scaling: float = 1.0,
    custom_map1: Optional[Callable[[ParameterExpression], ParameterExpression]] = None,
    custom_map2: Optional[
        Callable[[ParameterExpression, ParameterExpression], ParameterExpression]
    ] = None,
) -> QuantumCircuit:
    """Controlled‑modification polynomial ZZ feature map.

    Parameters
    ----------
    feature_dimension : int
        Number of qubits (must be ≥ 2).
    reps : int, default 2
        Number of repetitions of the feature‑map block.
    entanglement : str | Sequence | Callable, default "full"
        Specification of the two‑qubit entanglement pattern.
    single_coeffs : Sequence[float], default (1.0,)
        Coefficients for the single‑qubit polynomial φ₁(x) = Σ c_k · x^{k+1}.
        The same tuple is applied to all qubits.
    pair_weight : float, default 1.0
        Overall weight for the two‑qubit interaction φ₂(x, y) = w·(x·y + x²·y²).
    basis : str, default "h"
        Basis preparation: ``"h"`` applies Hadamards; ``"ry"`` applies RY(π/2).
    parameter_prefix : str, default "x"
        Prefix for the `ParameterVector` names.
    insert_barriers : bool, default False
        Insert barriers between logical blocks for visual clarity.
    name : str | None, default None
        Name of the resulting `QuantumCircuit`.  If ``None`` a default name is used.
    feature_scaling : float, default 1.0
        Multiplicative scaling factor applied to all input features before encoding.
    custom_map1 : Callable | None, default None
        Optional user‑supplied function for φ₁.  Must accept a `ParameterExpression`
        and return a `ParameterExpression`.
    custom_map2 : Callable | None, default None
        Optional user‑supplied function for φ₂.  Must accept two
        `ParameterExpression`s and return a `ParameterExpression`.

    Returns
    -------
    QuantumCircuit
        Parametrised circuit ready for binding with a classical feature vector.

    Raises
    ------
    ValueError
        If any of the input arguments are invalid.
    """
    if feature_dimension < 2:
        raise ValueError("feature_dimension must be >= 2.")
    if reps < 1:
        raise ValueError("reps must be >= 1.")
    if feature_scaling < 0:
        raise ValueError("feature_scaling must be non‑negative.")
    if pair_weight < 0:
        raise ValueError("pair_weight must be non‑negative.")
    if basis not in ("h", "ry"):
        raise ValueError("basis must be either 'h' or 'ry'.")

    n = int(feature_dimension)
    qc = QuantumCircuit(n, name=name or "ZZFeatureMapPolyControlledModification")

    # Parameter vector for classical data
    x = ParameterVector(parameter_prefix, n)

    # Default mapping functions (used if custom ones are not provided)
    def _map1(xi: ParameterExpression) -> ParameterExpression:
        """Polynomial φ₁(x) = Σ c_k · (s·x)^{k+1}."""
        scaled = xi * feature_scaling
        expr: ParameterExpression = 0
        power = scaled
        for coeff in single_coeffs:
            expr += coeff * power
            power *= scaled
        return expr

    def _map2(xi: ParameterExpression, xj: ParameterExpression) -> ParameterExpression:
        """Symmetric pair interaction φ₂(x, y) = w·(x·y + x²·y²)."""
        scaled_i = xi * feature_scaling
        scaled_j = xj * feature_scaling
        return pair_weight * (scaled_i * scaled_j + scaled_i**2 * scaled_j**2)

    # Use custom functions if provided
    map1 = custom_map1 if custom_map1 is not None else _map1
    map2 = custom_map2 if custom_map2 is not None else _map2

    # Resolve entanglement pattern
    pairs = _resolve_entanglement(n, entanglement)

    for rep in range(reps):
        # Basis preparation
        if basis == "h":
            qc.h(range(n))
        else:  # "ry"
            for q in range(n):
                qc.ry(pi / 2, q)
        if insert_barriers:
            qc.barrier()

        # Single‑qubit phases
        for i in range(n):
            qc.p(2 * map1(x[i]), i)
        if insert_barriers:
            qc.barrier()

        # Two‑qubit ZZ entanglement via CX–P–CX construction
        for (i, j) in pairs:
            angle = 2 * map2(x[i], x[j])
            qc.cx(i, j)
            qc.p(angle, j)
            qc.cx(i, j)

        if insert_barriers and rep!= reps - 1:
            qc.barrier()

    qc.input_params = x  # type: ignore[attr-defined]
    return qc


# ---------------------------------------------------------------------------
# Class wrapper
# ---------------------------------------------------------------------------

class ZZFeatureMapPolyControlledModification(QuantumCircuit):
    """QuantumCircuit subclass for the controlled‑modification polynomial ZZ feature map.

    The constructor forwards all arguments to
    :func:`zz_feature_map_poly_controlled_modification` and then composes the
    resulting circuit into the new instance.  The ``input_params`` attribute
    is preserved for parameter binding.

    Parameters
    ----------
    feature_dimension : int
    reps : int, default 2
    entanglement : str | Sequence | Callable, default "full"
    single_coeffs : Sequence[float], default (1.0,)
    pair_weight : float, default 1.0
    basis : str, default "h"
    parameter_prefix : str, default "x"
    insert_barriers : bool, default False
    name : str, default "ZZFeatureMapPolyControlledModification"
    feature_scaling : float, default 1.0
    custom_map1 : Callable | None, default None
    custom_map2 : Callable | None, default None
    """

    def __init__(
        self,
        feature_dimension: int,
        reps: int = 2,
        entanglement: Union[
            str,
            Sequence[Tuple[int, int]],
            Callable[[int], Sequence[Tuple[int, int]]],
        ] = "full",
        single_coeffs: Sequence[float] = (1.0,),
        pair_weight: float = 1.0,
        basis: str = "h",
        parameter_prefix: str = "x",
        insert_barriers: bool = False,
        name: str = "ZZFeatureMapPolyControlledModification",
        feature_scaling: float = 1.0,
        custom_map1: Optional[Callable[[ParameterExpression], ParameterExpression]] = None,
        custom_map2: Optional[
            Callable[[ParameterExpression, ParameterExpression], ParameterExpression]
        ] = None,
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
            name,
            feature_scaling,
            custom_map1,
            custom_map2,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)
        self.input_params = built.input_params  # type: ignore[attr-defined]


__all__ = [
    "ZZFeatureMapPolyControlledModification",
    "zz_feature_map_poly_controlled_modification",
]
