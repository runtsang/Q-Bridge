"""RealAmplitudesCZExtended – a depth‑controlled, hybrid‑entanglement RealAmplitudes variant.

This module extends the original RealAmplitudesCZ ansatz by:
* introducing an optional *mid‑entanglement* stage that can be toggled per repetition;
* allowing an optional *swap‑based* entanglement mechanism that inserts SWAP gates before each CZ;
* keeping the same rotation‑plus‑entanglement pattern for compatibility with existing workflows.
"""

from __future__ import annotations

from typing import Callable, List, Sequence, Tuple

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector


def _resolve_entanglement(
    num_qubits: int,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]],
) -> List[Tuple[int, int]]:
    """Return a list of two‑qubit pairs according to a simple entanglement spec."""
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


def _apply_entanglement(
    qc: QuantumCircuit,
    pairs: List[Tuple[int, int]],
    use_swaps: bool,
) -> None:
    """Apply CZ (or SWAP‑CZ) entanglement to the circuit."""
    for i, j in pairs:
        if use_swaps:
            qc.swap(i, j)
        qc.cz(i, j)


def real_amplitudes_cz_extended(
    num_qubits: int,
    reps: int = 1,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
    skip_final_rotation_layer: bool = False,
    insert_barriers: bool = False,
    mid_entanglement: bool = False,
    use_swaps: bool = False,
    parameter_prefix: str = "theta",
    name: str | None = None,
) -> QuantumCircuit:
    """
    Construct a depth‑controlled RealAmplitudes circuit with CZ entanglement.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    reps : int, default 1
        Number of repetition layers; each repetition consists of a rotation
        layer followed by an entanglement stage (and optionally a mid‑entanglement
        stage).
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]]
        Specification of two‑qubit pairs to entangle.  Accepted string values
        are ``"full"``, ``"linear"``, and ``"circular"``.  Custom lists or
        callables returning lists are also supported.
    skip_final_rotation_layer : bool, default False
        Whether to omit the final rotation layer after the last repetition.
    insert_barriers : bool, default False
        Insert barriers between logical blocks for easier debugging.
    mid_entanglement : bool, default False
        If ``True`` a second entanglement stage (identical to the first) is
        inserted after the first entanglement within each repetition.
    use_swaps : bool, default False
        If ``True`` a SWAP gate is applied before each CZ gate, effectively
        swapping the qubits before entanglement.  This can increase
        expressivity for certain hardware layouts.
    parameter_prefix : str, default "theta"
        Prefix for the rotation parameters.
    name : str | None, default None
        Circuit name; defaults to ``"RealAmplitudesCZExtended"``.

    Returns
    -------
    QuantumCircuit
        The constructed circuit.  The circuit exposes ``input_params`` and
        ``num_rot_layers`` attributes for convenient parameter binding.
    """
    if num_qubits < 1:
        raise ValueError("num_qubits must be >= 1.")
    if reps < 1:
        raise ValueError("reps must be >= 1.")

    n = int(num_qubits)
    qc = QuantumCircuit(n, name=name or "RealAmplitudesCZExtended")

    num_rot_layers = reps if skip_final_rotation_layer else reps + 1
    params = ParameterVector(parameter_prefix, num_rot_layers * n)

    def _rot(layer: int) -> None:
        base = layer * n
        for q in range(n):
            qc.ry(params[base + q], q)

    pairs = _resolve_entanglement(n, entanglement)

    for r in range(reps):
        _rot(r)
        if insert_barriers:
            qc.barrier()

        _apply_entanglement(qc, pairs, use_swaps)

        if insert_barriers:
            qc.barrier()

        if mid_entanglement:
            _apply_entanglement(qc, pairs, use_swaps)
            if insert_barriers:
                qc.barrier()

    if not skip_final_rotation_layer:
        _rot(reps)

    qc.input_params = params  # type: ignore[attr-defined]
    qc.num_rot_layers = num_rot_layers  # type: ignore[attr-defined]
    return qc


class RealAmplitudesCZExtended(QuantumCircuit):
    """Class wrapper for the extended CZ‑entangling RealAmplitudes ansatz."""

    def __init__(
        self,
        num_qubits: int,
        reps: int = 1,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
        skip_final_rotation_layer: bool = False,
        insert_barriers: bool = False,
        mid_entanglement: bool = False,
        use_swaps: bool = False,
        parameter_prefix: str = "theta",
        name: str = "RealAmplitudesCZExtended",
    ) -> None:
        built = real_amplitudes_cz_extended(
            num_qubits,
            reps,
            entanglement,
            skip_final_rotation_layer,
            insert_barriers,
            mid_entanglement,
            use_swaps,
            parameter_prefix,
            name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)
        self.input_params = built.input_params  # type: ignore[attr-defined]
        self.num_rot_layers = built.num_rot_layers  # type: ignore[attr-defined]


__all__ = ["RealAmplitudesCZExtended", "real_amplitudes_cz_extended"]
