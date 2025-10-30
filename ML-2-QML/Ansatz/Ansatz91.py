"""MirrorRealAmplitudes ansatz with mirror symmetry.

This module defines a parameterised ansatz that enforces a qubit‑reflection symmetry
across the rotation layers.  The rotation parameters for qubits *i* and *n‑1‑i*
are shared, cutting the number of free parameters roughly in half.  The
entanglement pattern and layer ordering are identical to the canonical
RealAmplitudes ansatz, but the symmetry constraint can be useful for problems
with left‑right invariance or when a reduced parameter set is desired.
"""

from __future__ import annotations

from typing import Callable, List, Sequence, Tuple, Union

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector


def _resolve_entanglement_pairs(
    num_qubits: int,
    entanglement: Union[
        str,
        Sequence[Tuple[int, int]],
        Callable[[int], Sequence[Tuple[int, int]]],
    ],
) -> List[Tuple[int, int]]:
    """
    Resolve a specification of two‑qubit entanglement pairs.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]]
        Entanglement specification.  Accepts the strings ``"full"``, ``"linear"``,
        and ``"circular"``, a user supplied list of pairs, or a callable that
        returns a list of pairs given ``num_qubits``.

    Returns
    -------
    List[Tuple[int, int]]
        A validated list of qubit pairs.

    Raises
    ------
    ValueError
        If an unknown string is provided, or a pair references an out‑of‑range
        qubit, or a pair connects a qubit to itself.
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
        pair_list = list(entanglement(num_qubits))
        return [(int(i), int(j)) for i, j in pair_list]

    pairs = [(int(i), int(j)) for i, j in entanglement]  # type: ignore[arg-type]
    for i, j in pairs:
        if i == j:
            raise ValueError("Entanglement pairs must connect distinct qubits.")
        if not (0 <= i < num_qubits and 0 <= j < num_qubits):
            raise ValueError(
                f"Entanglement pair {(i, j)} out of range for n={num_qubits}."
            )
    return pairs


def mirror_real_amplitudes(
    num_qubits: int,
    reps: int = 1,
    entanglement: Union[
        str, Sequence[Tuple[int, int]], Callable[[int], Sequence[Tuple[int, int]]]
    ] = "full",
    skip_final_rotation_layer: bool = False,
    insert_barriers: bool = False,
    parameter_prefix: str = "theta",
    name: str | None = None,
) -> QuantumCircuit:
    """Build a symmetry‑constrained RealAmplitudes circuit.

    The rotation angles for qubit *i* and *n‑1‑i* are shared across all
    rotation layers.  For odd numbers of qubits, the central qubit receives
    a unique parameter.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    reps : int, default=1
        Number of entanglement layers.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]], default="full"
        Specification of two‑qubit entanglement pairs.
    skip_final_rotation_layer : bool, default=False
        If True, the final rotation layer is omitted.
    insert_barriers : bool, default=False
        Insert barriers between layers for readability.
    parameter_prefix : str, default="theta"
        Prefix for the parameter vector.
    name : str | None, default=None
        Optional circuit name.

    Returns
    -------
    QuantumCircuit
        The constructed mirror‑symmetric ansatz.

    Raises
    ------
    ValueError
        If ``num_qubits`` is less than 1 or ``reps`` is negative.
    """
    if num_qubits < 1:
        raise ValueError("num_qubits must be >= 1.")
    if reps < 0:
        raise ValueError("reps must be non‑negative.")

    n = int(num_qubits)
    n_sym = (n + 1) // 2  # number of unique parameters per rotation layer
    num_rot_layers = reps if skip_final_rotation_layer else reps + 1
    params = ParameterVector(parameter_prefix, num_rot_layers * n_sym)

    def _rotation_layer(layer_idx: int) -> None:
        base = layer_idx * n_sym
        for q in range(n):
            mirror_idx = min(q, n - 1 - q)
            qc.ry(params[base + mirror_idx], q)

    qc = QuantumCircuit(n, name=name or "MirrorRealAmplitudes")
    pairs = _resolve_entanglement_pairs(n, entanglement)

    for r in range(reps):
        _rotation_layer(r)
        if insert_barriers:
            qc.barrier()
        for i, j in pairs:
            qc.cx(i, j)
        if insert_barriers:
            qc.barrier()

    if not skip_final_rotation_layer:
        _rotation_layer(reps)

    qc.input_params = params  # type: ignore[attr-defined]
    qc.num_rot_layers = num_rot_layers  # type: ignore[attr-defined]
    return qc


class MirrorRealAmplitudes(QuantumCircuit):
    """Class‑style wrapper for the mirror‑symmetric RealAmplitudes ansatz.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    reps : int, default=1
        Number of entanglement layers.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]], default="full"
        Specification of two‑qubit entanglement pairs.
    skip_final_rotation_layer : bool, default=False
        If True, the final rotation layer is omitted.
    insert_barriers : bool, default=False
        Insert barriers between layers.
    parameter_prefix : str, default="theta"
        Prefix for the parameter vector.
    name : str, default="MirrorRealAmplitudes"
        Circuit name.

    Attributes
    ----------
    input_params : ParameterVector
        Parameter vector used in the ansatz.
    num_rot_layers : int
        Number of rotation layers present.
    """

    def __init__(
        self,
        num_qubits: int,
        reps: int = 1,
        entanglement: Union[
            str, Sequence[Tuple[int, int]], Callable[[int], Sequence[Tuple[int, int]]]
        ] = "full",
        skip_final_rotation_layer: bool = False,
        insert_barriers: bool = False,
        parameter_prefix: str = "theta",
        name: str = "MirrorRealAmplitudes",
    ) -> None:
        built = mirror_real_amplitudes(
            num_qubits=num_qubits,
            reps=reps,
            entanglement=entanglement,
            skip_final_rotation_layer=skip_final_rotation_layer,
            insert_barriers=insert_barriers,
            parameter_prefix=parameter_prefix,
            name=name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)

        self.input_params = built.input_params  # type: ignore[attr-defined]
        self.num_rot_layers = built.num_rot_layers  # type: ignore[attr-defined]


__all__ = ["MirrorRealAmplitudes", "mirror_real_amplitudes"]
