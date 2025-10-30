"""ZZFeatureMapRZZExtension module.

This module implements a Qiskit‑compatible feature‑map circuit that builds on the
original ZZFeatureMapRZZ.  It introduces three main extensions:

* **Higher‑order interactions** – 3‑qubit and 4‑qubit interaction terms are
  encoded by additional RZZ layers whose angles are derived from products of
  three or four data parameters.  The user can control the relative strength
  via *triple_scale* and *quadruple_scale*.
* **Pre‑ and post‑rotations** – optional single‑qubit Z rotations before
  the entangling layer and after the final layer.  By default these are no‑ops
  but can be supplied by the user to bias the rotation angles.
* **Data rescaling** – a user‑supplied *data_map_func* can transform the raw
  feature vector before it is mapped to gate angles.  If omitted, the default
  linear mapping is used.

The interface mirrors the seed module so that existing code can switch to
the upgraded map with minimal changes.

---

### Usage example

```python
from zz_feature_map_rzz_extension import zz_feature_map_rzz_extension

qc = zz_feature_map_rzz_extension(
    feature_dimension=4,
    reps=3,
    entanglement="circular",
    data_map_func=lambda x: [xi * 0.5 for xi in x],  # simple scaling
    pre_rotation_func=lambda x: [0.0] * len(x),       # no pre‑rotation
    post_rotation_func=lambda x: [0.0] * len(x),      # no post‑rotation
    pair_scale=1.0,
    triple_scale=0.5,
    quadruple_scale=0.25,
    insert_barriers=True,
)

# Bind parameters:
import numpy as np
params = np.array([0.1, 0.2, 0.3, 0.4])
bound_qc = qc.bind_parameters(zip(qc.input_params, params))
```

"""

from __future__ import annotations

import math
from typing import Callable, List, Tuple, Sequence

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterExpression, ParameterVector


# ----------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------


def _resolve_entanglement(
    num_qubits: int,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]],
) -> List[Tuple[int, int]]:
    """Return a list of two‑qubit pairs.

    Supported specs:
      * ``"full"``     – all-to-all pairs (i < j)
      * ``"linear"``   – nearest‑neighbor chain
      * ``"circular"`` – linear chain plus wrap‑around (n‑1,0)
      * explicit list of pairs
      * callable: f(n) -> sequence of (i, j)
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


# ----------------------------------------------------------------------
# Default data mapping functions
# ----------------------------------------------------------------------


def _default_map_1(x: ParameterExpression) -> ParameterExpression:
    """Default φ1(x) = x."""
    return x


def _default_map_2(x: ParameterExpression, y: ParameterExpression) -> ParameterExpression:
    """Default φ2(x, y) = (π − x)(π − y)."""
    return (math.pi - x) * (math.pi - y)


def _default_map_3(x: ParameterExpression, y: ParameterExpression, z: ParameterExpression) -> ParameterExpression:
    """Default φ3(x, y, z) = (π − x)(π − y)(π − z)."""
    return (math.pi - x) * (math.pi - y) * (math.pi - z)


def _default_map_4(a: ParameterExpression, b: ParameterExpression, c: ParameterExpression, d: ParameterExpression) -> ParameterExpression:
    """Default φ4(a, b, c, d) = (π − a)(π − b)(π − c)(π − d)."""
    return (math.pi - a) * (math.pi - b) * (math.pi - c) * (math.pi - d)


# ----------------------------------------------------------------------
# Feature‑map construction
# ----------------------------------------------------------------------


def zz_feature_map_rzz_extension(
    feature_dimension: int,
    reps: int = 2,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
    data_map_func: Callable[[Sequence[ParameterExpression]], ParameterExpression] | None = None,
    pre_rotation_func: Callable[[Sequence[ParameterExpression]], Sequence[ParameterExpression]] | None = None,
    post_rotation_func: Callable[[Sequence[ParameterExpression]], Sequence[ParameterExpression]] | None = None,
    pair_scale: float = 1.0,
    triple_scale: float = 0.5,
    quadruple_scale: float = 0.25,
    parameter_prefix: str = "x",
    insert_barriers: bool = False,
    name: str | None = None,
) -> QuantumCircuit:
    """Construct a ZZ‑based feature map with extended interactions.

    Parameters
    ----------
    feature_dimension : int
        Number of classical features / qubits. Must be >= 2.
    reps : int, default: 2
        Number of repetitions of the encoding block.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]]
        Entanglement pattern. See _resolve_entanglement for supported values.
    data_map_func : Callable[[Sequence[ParameterExpression]], ParameterExpression] | None
        Function that maps a list of parameters to a single rotation angle.
        If ``None`` the defaults below are used.
    pre_rotation_func : Callable[[Sequence[ParameterExpression]], Sequence[ParameterExpression]] | None
        Function that returns a list of Z‑rotation angles applied before the
        entangling layer.  If ``None`` no pre‑rotation is applied.
    post_rotation_func : Callable[[Sequence[ParameterExpression]], Sequence[ParameterExpression]] | None
        Function that returns a list of Z‑rotation angles applied after the
        entangling layer.  If ``None`` no post‑rotation is applied.
    pair_scale : float, default: 1.0
        Scaling factor applied to the pair‑interaction angles.
    triple_scale : float, default: 0.5
        Scaling factor applied to the triple‑interaction angles.
    quadruple_scale : float, default: 0.25
        Scaling factor applied to the quadruple‑interaction angles.
    parameter_prefix : str, default: "x"
        Prefix for the automatic ParameterVector.
    insert_barriers : bool, default: False
        If ``True`` a barrier is inserted after each major block.
    name : str | None
        Optional circuit name.  If ``None`` a default is used.

    Returns
    -------
    QuantumCircuit
        The constructed feature‑map circuit.  The circuit has an attribute
        ``input_params`` that holds the ParameterVector for binding.

    Raises
    ------
    ValueError
        If *feature_dimension* < 2 or if the entanglement specification is
        invalid.
    """
    if feature_dimension < 2:
        raise ValueError("feature_dimension must be >= 2.")
    n = int(feature_dimension)
    qc = QuantumCircuit(n, name=name or "ZZFeatureMapRZZExtension")

    # Parameter vector for the raw features
    x = ParameterVector(parameter_prefix, n)

    # Default mapping functions if none supplied
    if data_map_func is None:
        map1 = _default_map_1
        map2 = _default_map_2
        map3 = _default_map_3
        map4 = _default_map_4
    else:
        # Wrap user function to accept variable number of args
        def _wrap(func: Callable[[Sequence[ParameterExpression]], ParameterExpression]):
            return lambda *args: func(args)

        map1 = lambda xi: _wrap(data_map_func)(xi)
        map2 = lambda xi, xj: _wrap(data_map_func)(xi, xj)
        map3 = lambda xi, xj, xk: _wrap(data_map_func)(xi, xj, xk)
        map4 = lambda a, b, c, d: _wrap(data_map_func)(a, b, c, d)

    pairs = _resolve_entanglement(n, entanglement)

    # Pre‑rotation angles
    if pre_rotation_func is None:
        pre_rot_angles = [0.0] * n
    else:
        pre_rot_angles = pre_rotation_func(x)

    # Post‑rotation angles
    if post_rotation_func is None:
        post_rot_angles = [0.0] * n
    else:
        post_rot_angles = post_rotation_func(x)

    for rep in range(int(reps)):
        # Pre‑rotations
        for i, ang in enumerate(pre_rot_angles):
            qc.p(ang, i)
        if insert_barriers:
            qc.barrier()

        # First Hadamard layer
        qc.h(range(n))

        # Single‑qubit phase rotations
        for i in range(n):
            qc.p(2 * map1(x[i]), i)

        # Pairwise RZZ entanglement
        for (i, j) in pairs:
            qc.rzz(2 * pair_scale * map2(x[i], x[j]), i, j)

        # Triple‑qubit interaction via additional RZZ layers
        # (approximate higher‑order term by weighting pairwise gates)
        for (i, j) in pairs:
            for k in range(n):
                if k in (i, j):
                    continue
                qc.rzz(2 * triple_scale * map3(x[i], x[j], x[k]), i, j)

        # Quadruple‑qubit interaction via pairwise weighting
        for (i, j) in pairs:
            for k in range(n):
                for l in range(k + 1, n):
                    if len({i, j, k, l}) < 4:
                        continue
                    qc.rzz(2 * quadruple_scale * map4(x[i], x[j], x[k], x[l]), i, j)

        # Post‑rotations
        for i, ang in enumerate(post_rot_angles):
            qc.p(ang, i)

        if insert_barriers and rep!= reps - 1:
            qc.barrier()

    qc.input_params = x  # type: ignore[attr-defined]
    return qc


# ----------------------------------------------------------------------
# Classical subclass
# ----------------------------------------------------------------------


class ZZFeatureMapRZZExtension(QuantumCircuit):
    """Object‑oriented wrapper for the extended ZZ‑RZZ feature map.

    The constructor forwards all arguments to :func:`zz_feature_map_rzz_extension`
    and composes the resulting circuit into ``self``.
    """

    def __init__(
        self,
        feature_dimension: int,
        reps: int = 2,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
        data_map_func: Callable[[Sequence[ParameterExpression]], ParameterExpression] | None = None,
        pre_rotation_func: Callable[[Sequence[ParameterExpression]], Sequence[ParameterExpression]] | None = None,
        post_rotation_func: Callable[[Sequence[ParameterExpression]], Sequence[ParameterExpression]] | None = None,
        pair_scale: float = 1.0,
        triple_scale: float = 0.5,
        quadruple_scale: float = 0.25,
        parameter_prefix: str = "x",
        insert_barriers: bool = False,
        name: str = "ZZFeatureMapRZZExtension",
    ) -> None:
        built = zz_feature_map_rzz_extension(
            feature_dimension,
            reps,
            entanglement,
            data_map_func,
            pre_rotation_func,
            post_rotation_func,
            pair_scale,
            triple_scale,
            quadruple_scale,
            parameter_prefix,
            insert_barriers,
            name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)
        self.input_params = built.input_params  # type: ignore[attr-defined]


__all__ = ["ZZFeatureMapRZZExtension", "zz_feature_map_rzz_extension"]
