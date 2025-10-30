"""Extended polynomial ZZFeatureMap with higher‑order interactions and optional pre/post rotations."""

from __future__ import annotations

from math import pi
from typing import Callable, Iterable, List, Sequence, Tuple, Dict, Optional

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterExpression, ParameterVector


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _resolve_entanglement(
    num_qubits: int,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]],
) -> List[Tuple[int, int]]:
    """Return a list of two‑qubit pairs according to a simple entanglement spec.

    Supported specs:
      - "full": all‑to‑all pairs (i < j)
      - "linear": nearest neighbors (0,1), (1,2), …
      - "circular": linear plus wrap‑around (n‑1,0) if n > 2
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
    # basic validation
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
    """Default φ2(x, y) = x · y."""
    return x * y


def _default_map_3(x: ParameterExpression, y: ParameterExpression, z: ParameterExpression) -> ParameterExpression:
    """Default φ3(x, y, z) = x · y · z."""
    return x * y * z


def normalize_features(features: Sequence[float]) -> List[float]:
    """Return a unit‑norm copy of *features*.

    This helper can be used to pre‑scale classical data before it is
    bound to the circuit parameters.  The function is safe to call even
    when the input is already normalised.

    Parameters
    ----------
    features : Sequence[float]
        Classical feature vector.

    Returns
    -------
    List[float]
        Normalised feature vector.
    """
    import math

    norm = math.sqrt(sum(f * f for f in features))
    if norm == 0:
        raise ValueError("Cannot normalise a zero‑norm feature vector.")
    return [float(f / norm) for f in features]


# ---------------------------------------------------------------------------
# Feature‑map construction
# ---------------------------------------------------------------------------

def zz_feature_map_poly_extended(
    feature_dimension: int,
    reps: int = 2,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
    single_coeffs: Sequence[float] = (1.0,),
    pair_weight: float = 1.0,
    triple_weight: float = 1.0,
    basis: str = "h",  # "h" or "ry"
    parameter_prefix: str = "x",
    pre_rotation: bool = False,
    pre_rotation_angle: float = pi / 4,
    post_rotation: bool = False,
    post_rotation_angle: float = pi / 4,
    insert_barriers: bool = False,
    name: str | None = None,
) -> QuantumCircuit:
    """
    Polynomial ZZ feature map with support for single‑, pair‑ and triple‑body interactions.
    The circuit is compatible with Qiskit's data‑encoding workflows and can be used
    directly as a sub‑circuit in variational algorithms.

    Parameters
    ----------
    feature_dimension : int
        Number of classical features and qubits.
    reps : int, default 2
        Number of repetitions (depth) of the feature‑map layer.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]]
        Entanglement pattern for pair interactions.
    single_coeffs : Sequence[float], default (1.0,)
        Coefficients for the polynomial φ1(x) = Σ c_k · x^{k+1}.
    pair_weight : float, default 1.0
        Overall weight for pair interactions φ2(x, y) = pair_weight · x · y.
    triple_weight : float, default 1.0
        Overall weight for triple interactions φ3(x, y, z) = triple_weight · x · y · z.
    basis : str, default "h"
        Basis preparation: "h" for Hadamard, "ry" for RY(π/2).
    parameter_prefix : str, default "x"
        Prefix for the ParameterVector.
    pre_rotation : bool, default False
        If True, prepend a RY(pre_rotation_angle) on every qubit before basis preparation.
    pre_rotation_angle : float, default π/4
        Angle for the optional pre‑rotation.
    post_rotation : bool, default False
        If True, append a RY(post_rotation_angle) on every qubit after entanglement.
    post_rotation_angle : float, default π/4
        Angle for the optional post‑rotation.
    insert_barriers : bool, default False
        Insert barriers between logical sections for easier debugging.
    name : str | None, default None
        Name of the resulting circuit.

    Returns
    -------
    QuantumCircuit
        Parameterised circuit ready for data binding.

    Notes
    -----
    * The function raises a ValueError if ``feature_dimension`` is less than 2
      because pair interactions require at least two qubits.
    * The circuit exposes an ``input_params`` attribute containing the
      ParameterVector used for data binding.
    * For higher‑order interactions beyond triples, the user must extend
      the implementation manually.
    """
    if feature_dimension < 2:
        raise ValueError("feature_dimension must be >= 2 to support pair interactions.")
    n = int(feature_dimension)

    qc = QuantumCircuit(n, name=name or "ZZFeatureMapPolyExtended")

    # Parameter vector for classical data
    x = ParameterVector(parameter_prefix, n)

    # Mapping functions
    def map1(xi: ParameterExpression) -> ParameterExpression:
        expr = 0
        p = xi
        for c in single_coeffs:
            expr = expr + c * p
            p = p * xi
        return expr

    def map2(xi: ParameterExpression, xj: ParameterExpression) -> ParameterExpression:
        return pair_weight * xi * xj

    def map3(xi: ParameterExpression, xj: ParameterExpression, xk: ParameterExpression) -> ParameterExpression:
        return triple_weight * xi * xj * xk

    pairs = _resolve_entanglement(n, entanglement)

    for rep in range(int(reps)):
        # Optional pre‑rotation
        if pre_rotation:
            for q in range(n):
                qc.ry(pre_rotation_angle, q)

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

        # Pair ZZ interactions
        for (i, j) in pairs:
            angle = 2 * map2(x[i], x[j])
            qc.cx(i, j)
            qc.p(angle, j)
            qc.cx(i, j)

        # Triple ZZ interactions (only if n >= 3)
        if n >= 3:
            # Generate all distinct triples (i < j < k)
            for i in range(n - 2):
                for j in range(i + 1, n - 1):
                    for k in range(j + 1, n):
                        angle = 2 * map3(x[i], x[j], x[k])
                        qc.cx(i, j)
                        qc.cx(i, k)
                        qc.p(angle, k)
                        qc.cx(i, k)
                        qc.cx(i, j)

        # Optional post‑rotation
        if post_rotation:
            for q in range(n):
                qc.ry(post_rotation_angle, q)

        if insert_barriers and rep!= reps - 1:
            qc.barrier()

    # expose parameter vector for easy binding
    qc.input_params = x  # type: ignore[attr-defined]
    return qc


class ZZFeatureMapPolyExtended(QuantumCircuit):
    """Object‑oriented wrapper for the extended polynomial ZZ feature map.

    The constructor accepts the same arguments as :func:`zz_feature_map_poly_extended`
    and builds a circuit that can be used directly in variational circuits.

    Example
    -------
    >>> from qiskit.circuit import ParameterVector
    >>> from qiskit import Aer, transpile
    >>> from qiskit.utils import QuantumInstance
    >>> from qiskit.algorithms import VQE
    >>> from qiskit.opflow import PauliSumOp, Z, Y, X
    >>> from qiskit_nature.drivers import PySCFDriver, UnitsType
    >>> from qiskit_nature.problems.second_quantization import ElectronicStructureProblem
    >>> from qiskit_nature.converters.second_quantization import QubitConverter
    >>> from qiskit_nature.mappers.second_quantization import JordanWignerMapper
    >>> from qiskit_nature.algorithms import VQEUCCFactory
    >>> from qiskit_nature.algorithms import VQE
    >>> from qiskit_nature.algorithms import VQEUCCFactory
    >>> from qiskit_nature.algorithms import VQEUCC
    >>> # The example is illustrative; actual usage requires integration with a chemistry backend.
    """

    def __init__(
        self,
        feature_dimension: int,
        reps: int = 2,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
        single_coeffs: Sequence[float] = (1.0,),
        pair_weight: float = 1.0,
        triple_weight: float = 1.0,
        basis: str = "h",
        parameter_prefix: str = "x",
        pre_rotation: bool = False,
        pre_rotation_angle: float = pi / 4,
        post_rotation: bool = False,
        post_rotation_angle: float = pi / 4,
        insert_barriers: bool = False,
        name: str = "ZZFeatureMapPolyExtended",
    ) -> None:
        built = zz_feature_map_poly_extended(
            feature_dimension,
            reps,
            entanglement,
            single_coeffs,
            pair_weight,
            triple_weight,
            basis,
            parameter_prefix,
            pre_rotation,
            pre_rotation_angle,
            post_rotation,
            post_rotation_angle,
            insert_barriers,
            name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)
        self.input_params = built.input_params  # type: ignore[attr-defined]


__all__ = ["ZZFeatureMapPolyExtended", "zz_feature_map_poly_extended", "normalize_features"]
