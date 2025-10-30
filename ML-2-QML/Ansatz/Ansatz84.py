"""RealAmplitudesAlternatingExtended: a deeper, more expressive ansatz.

This module defines a parameterised quantum circuit that
- alternates between RY and RX rotations on each qubit,
- inserts a *controlled‑RZ* block after every rotation layer,
- supports a user‑supplied entanglement schedule (string or callable),
- optionally applies a global‑phase RZ gate,
- and can insert barriers for debugging or compilation control.

The design keeps the intuition of the seed while adding depth and flexibility.
"""

from __future__ import annotations

from typing import Callable, Iterable, List, Sequence, Tuple

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector


def _entanglement_pairs(
    num_qubits: int,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Iterable[Tuple[int, int]]],
) -> List[Tuple[int, int]]:
    """
    Resolve a list of two‑qubit entanglement pairs.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    entanglement : str | Sequence[Tuple[int, int]] | Callable
        * ``"full"``  – all possible pairs.
        * ``"linear"`` – nearest‑neighbour chain.
        * ``"circular"`` – linear plus a connection between the last and first qubit.
        * ``"none"`` – no entanglement.
        * Callable – must return an iterable of distinct qubit pairs.

    Returns
    -------
    List[Tuple[int, int]]
        Validated list of qubit pairs.

    Raises
    ------
    ValueError
        If the specification is unknown or contains invalid pairs.
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
        if entanglement == "none":
            return []
        raise ValueError(f"Unknown entanglement string: {entanglement!r}")

    if callable(entanglement):
        pairs = list(entanglement(num_qubits))
        return [(int(i), int(j)) for i, j in pairs]

    pairs = [(int(i), int(j)) for i, j in entanglement]  # type: ignore[arg-type]
    for i, j in pairs:
        if i == j:
            raise ValueError("Entanglement pairs must connect distinct qubits.")
        if not (0 <= i < num_qubits and 0 <= j < num_qubits):
            raise ValueError(f"Entanglement pair {(i, j)} out of range for n={num_qubits}.")
    return pairs


def real_amplitudes_alternating_extended(
    num_qubits: int,
    reps: int = 1,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Iterable[Tuple[int, int]]] = "full",
    skip_final_rotation_layer: bool = False,
    insert_barriers: bool = False,
    parameter_prefix: str = "theta",
    global_phase: bool = False,
    name: str | None = None,
) -> QuantumCircuit:
    """
    Construct an extended Real‑Amplitudes ansatz with alternating rotations and
    controlled‑RZ layers.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    reps : int, default 1
        Number of rotation‑entanglement blocks.
    entanglement : str | Sequence[Tuple[int, int]] | Callable, default "full"
        Entanglement schedule.
    skip_final_rotation_layer : bool, default False
        If True, omit the last rotation layer that normally follows the final entanglement.
    insert_barriers : bool, default False
        Insert a barrier after each logical block for easier debugging.
    parameter_prefix : str, default "theta"
        Prefix for the parameter names.
    global_phase : bool, default False
        If True, add a single global RZ gate applied to all qubits at the end.
    name : str | None, default None
        Circuit name.

    Returns
    -------
    QuantumCircuit
        Parameterised circuit ready for composition or execution.
    """
    if num_qubits < 1:
        raise ValueError("num_qubits must be >= 1.")
    if reps < 0:
        raise ValueError("reps must be non‑negative.")

    n = int(num_qubits)
    qc = QuantumCircuit(n, name=name or "RealAmplitudesAlternatingExtended")

    pairs = _entanglement_pairs(n, entanglement)

    num_rot_layers = reps if skip_final_rotation_layer else reps + 1
    num_cz_pairs = len(pairs)

    # Parameter vectors
    params_rot = ParameterVector(parameter_prefix + "_rot", num_rot_layers * n)
    params_cz = ParameterVector(parameter_prefix + "_cz", num_rot_layers * num_cz_pairs)
    global_param = ParameterVector(parameter_prefix + "_global", 1) if global_phase else None

    def _rot(layer: int) -> None:
        """Apply a layer of RY or RX rotations."""
        base = layer * n
        if layer % 2 == 0:
            for q in range(n):
                qc.ry(params_rot[base + q], q)
        else:
            for q in range(n):
                qc.rx(params_rot[base + q], q)

    def _c_rz(layer: int) -> None:
        """Apply a controlled‑RZ block for the given layer."""
        base = layer * num_cz_pairs
        for idx, (i, j) in enumerate(pairs):
            qc.crz(params_cz[base + idx], i, j)

    for r in range(reps):
        _rot(r)
        if insert_barriers:
            qc.barrier()
        _c_rz(r)
        if insert_barriers:
            qc.barrier()
        for i, j in pairs:
            qc.cx(i, j)
        if insert_barriers:
            qc.barrier()

    if not skip_final_rotation_layer:
        _rot(reps)
        if insert_barriers:
            qc.barrier()

    if global_phase:
        # Global RZ applied to all qubits with the same parameter
        for q in range(n):
            qc.rz(global_param[0], q)

    # Attach metadata for external tooling
    qc.input_params = ParameterVector(parameter_prefix, num_rot_layers * (n + num_cz_pairs) + (1 if global_phase else 0))
    qc.num_rot_layers = num_rot_layers
    qc.num_entanglement_pairs = num_cz_pairs
    qc.global_phase_param = global_param

    return qc


class RealAmplitudesAlternatingExtended(QuantumCircuit):
    """Convenience subclass exposing the extended ansatz as a QuantumCircuit."""

    def __init__(
        self,
        num_qubits: int,
        reps: int = 1,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Iterable[Tuple[int, int]]] = "full",
        skip_final_rotation_layer: bool = False,
        insert_barriers: bool = False,
        parameter_prefix: str = "theta",
        global_phase: bool = False,
        name: str = "RealAmplitudesAlternatingExtended",
    ) -> None:
        built = real_amplitudes_alternating_extended(
            num_qubits,
            reps,
            entanglement,
            skip_final_rotation_layer,
            insert_barriers,
            parameter_prefix,
            global_phase,
            name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)
        self.input_params = built.input_params  # type: ignore[attr-defined]
        self.num_rot_layers = built.num_rot_layers  # type: ignore[attr-defined]
        self.num_entanglement_pairs = built.num_entanglement_pairs  # type: ignore[attr-defined]
        self.global_phase_param = built.global_phase_param  # type: ignore[attr-defined]


__all__ = ["RealAmplitudesAlternatingExtended", "real_amplitudes_alternating_extended"]
