"""
RealAmplitudes Alternating with Controlled Symmetry
===================================================

This module implements a *controlled‑modification* of the original
`RealAmplitudesAlternating` ansatz.  The new ansatz keeps the
alternating rotation pattern but introduces **parameter sharing** between
the two rotation subsets (RY‑ and RX‑layers).  The sharing enforces
a reflection symmetry across the qubit array, which is useful for
datasets that require invariant features.  The implementation is
fully‑compatible with Qiskit, and exposes both a convenience
function and a subclass of `QuantumCircuit` with the same name.
"""

from __future__ import annotations

from typing import Callable, Iterable, List, Sequence, Tuple, Union

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector


def _resolve_entanglement(
    num_qubits: int,
    entanglement: Union[
        str,
        Sequence[Tuple[int, int]],
        Callable[[int], Sequence[Tuple[int, int]]],
    ],
) -> List[Tuple[int, int]]:
    """
    Resolve the entanglement specification into a list of two‑qubit pairs.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]]
        Entanglement pattern.  Supported string values are ``"full"``,
        ``"linear"``, and ``"circular"``.  A custom sequence or callable
        can be supplied to specify arbitrary pairs.

    Returns
    -------
    List[Tuple[int, int]]
        A validated list of distinct qubit pairs.

    Raises
    ------
    ValueError
        If an invalid string or pair is provided.
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


def real_amplitudes_alternating_controlled(
    num_qubits: int,
    reps: int = 1,
    entanglement: Union[
        str,
        Sequence[Tuple[int, int]],
        Callable[[int], Sequence[Tuple[int, int]]],
    ] = "full",
    skip_final_rotation_layer: bool = False,
    insert_barriers: bool = False,
    parameter_prefix: str = "theta",
    name: str | None = None,
) -> QuantumCircuit:
    """
    Build a RealAmplitudes alternating ansatz with symmetry‑controlled
    parameter sharing.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    reps : int, default 1
        Number of entanglement layers.  The circuit will contain
        ``reps`` entanglement blocks and either ``reps`` or ``reps + 1``
        rotation layers depending on ``skip_final_rotation_layer``.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]], default "full"
        Entanglement specification.
    skip_final_rotation_layer : bool, default False
        If ``True``, the final rotation layer is omitted.
    insert_barriers : bool, default False
        If ``True``, a barrier is inserted after each rotation and
        entanglement block.
    parameter_prefix : str, default "theta"
        Prefix for the parameters in the ``ParameterVector``.
    name : str | None, default None
        Optional name for the resulting ``QuantumCircuit``.  If ``None``,
        ``"RealAmplitudesAlternatingControlled"`` is used.

    Returns
    -------
    QuantumCircuit
        The constructed ansatz circuit.

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
    qc = QuantumCircuit(n, name=name or "RealAmplitudesAlternatingControlled")

    # Number of rotation layers: one per entanglement block plus an optional final layer.
    num_rot_layers = reps if skip_final_rotation_layer else reps + 1
    # Each rotation layer shares parameters across symmetric qubit pairs.
    half = (n + 1) // 2  # ceil(n/2)
    params = ParameterVector(parameter_prefix, num_rot_layers * half)

    def _rot(layer: int) -> None:
        """Apply the alternating rotation layer with symmetry sharing."""
        base = layer * half
        for q in range(n):
            # Mirror index for symmetry: use the smaller of q and n-1-q.
            idx = min(q, n - 1 - q)
            param = params[base + idx]
            if layer % 2 == 0:
                qc.ry(param, q)
            else:
                qc.rx(param, q)

    pairs = _resolve_entanglement(n, entanglement)

    for r in range(reps):
        _rot(r)
        if insert_barriers:
            qc.barrier()
        for (i, j) in pairs:
            qc.cx(i, j)
        if insert_barriers:
            qc.barrier()

    if not skip_final_rotation_layer:
        _rot(reps)

    qc.input_params = params  # type: ignore[attr-defined]
    qc.num_rot_layers = num_rot_layers  # type: ignore[attr-defined]
    return qc


class RealAmplitudesAlternatingControlled(QuantumCircuit):
    """Convenience subclass of :class:`~qiskit.QuantumCircuit` for the
    symmetry‑controlled alternating ansatz.

    Parameters
    ----------
    num_qubits : int
        Number of qubits.
    reps : int, default 1
        Number of entanglement layers.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]], default "full"
        Entanglement specification.
    skip_final_rotation_layer : bool, default False
        Skip the final rotation layer if ``True``.
    insert_barriers : bool, default False
        Insert a barrier after each block if ``True``.
    parameter_prefix : str, default "theta"
        Prefix for the parameters.
    name : str, default "RealAmplitudesAlternatingControlled"
        Circuit name.
    """

    def __init__(
        self,
        num_qubits: int,
        reps: int = 1,
        entanglement: Union[
            str,
            Sequence[Tuple[int, int]],
            Callable[[int], Sequence[Tuple[int, int]]],
        ] = "full",
        skip_final_rotation_layer: bool = False,
        insert_barriers: bool = False,
        parameter_prefix: str = "theta",
        name: str = "RealAmplitudesAlternatingControlled",
    ) -> None:
        built = real_amplitudes_alternating_controlled(
            num_qubits,
            reps,
            entanglement,
            skip_final_rotation_layer,
            insert_barriers,
            parameter_prefix,
            name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)
        self.input_params = built.input_params  # type: ignore[attr-defined]
        self.num_rot_layers = built.num_rot_layers  # type: ignore[attr-defined]


__all__ = [
    "RealAmplitudesAlternatingControlled",
    "real_amplitudes_alternating_controlled",
]
