"""
RealAmplitudesAlternatingExtended: A depth‑controlled, hybrid‑phase extension of the original alternating‑rotation RealAmplitudes ansatz.
"""

from __future__ import annotations

import math
from typing import Callable, Iterable, List, Sequence, Tuple

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

__all__ = ["RealAmplitudesAlternatingExtended", "real_amplitudes_alternating_extended"]


def _resolve_entanglement(
    num_qubits: int,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Iterable[Tuple[int, int]]],
    max_depth: int,
) -> List[Tuple[int, int]]:
    """
    Resolve the entanglement specification into a list of two‑qubit pairs.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Iterable[Tuple[int, int]]]
        Either a string specifier, a custom list of pairs, or a callable that returns pairs.
    max_depth : int
        Maximum number of entanglement pairs to use.  This limits the graph depth
        and can reduce the number of CX gates when the full graph would be too dense.

    Returns
    -------
    List[Tuple[int, int]]
        A list of qubit pairs (i, j) to entangle with CX gates.

    Raises
    ------
    ValueError
        If an invalid specification is provided or a pair references out‑of‑range qubits.
    """
    if isinstance(entanglement, str):
        if entanglement == "full":
            pairs = [(i, j) for i in range(num_qubits) for j in range(i + 1, num_qubits)]
        elif entanglement == "linear":
            pairs = [(i, i + 1) for i in range(num_qubits - 1)]
        elif entanglement == "circular":
            pairs = [(i, i + 1) for i in range(num_qubits - 1)]
            if num_qubits > 2:
                pairs.append((num_qubits - 1, 0))
        else:
            raise ValueError(f"Unknown entanglement string: {entanglement!r}")
    elif callable(entanglement):
        pairs = list(entanglement(num_qubits))
    else:
        pairs = [(int(i), int(j)) for (i, j) in entanglement]  # type: ignore[arg-type]

    # Validate pairs
    for i, j in pairs:
        if i == j:
            raise ValueError("Entanglement pairs must connect distinct qubits.")
        if not (0 <= i < num_qubits and 0 <= j < num_qubits):
            raise ValueError(f"Entanglement pair {(i, j)} out of range for n={num_qubits}.")

    # Truncate to max_depth
    if max_depth is not None and len(pairs) > max_depth:
        pairs = pairs[:max_depth]
    return pairs


def real_amplitudes_alternating_extended(
    num_qubits: int,
    reps: int = 1,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Iterable[Tuple[int, int]]] = "full",
    max_entanglement_depth: int | None = None,
    skip_final_rotation_layer: bool = False,
    insert_barriers: bool = False,
    parameter_prefix: str = "theta",
    phase_parameter_prefix: str = "phi",
    name: str | None = None,
) -> QuantumCircuit:
    """
    Build a depth‑controlled, hybrid‑phase RealAmplitudes ansatz with alternating RX/RY rotations.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    reps : int, default 1
        Number of alternating rotation layers.  If ``skip_final_rotation_layer`` is False,
        an additional rotation layer is added after the last entanglement block.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Iterable[Tuple[int, int]]], default "full"
        Specification of the entanglement pattern.  Strings "full", "linear", and "circular" are
        supported.  Custom sequences or callables are also accepted.
    max_entanglement_depth : int | None, default None
        Maximum number of entanglement pairs to use per layer.  A value of ``None`` uses the
        full set derived from ``entanglement``.  This parameter allows quick depth tuning.
    skip_final_rotation_layer : bool, default False
        If True, the final rotation layer after the last entanglement block is omitted.
    insert_barriers : bool, default False
        If True, insert barriers before and after each entanglement block for easier visual
        inspection and debugging.
    parameter_prefix : str, default "theta"
        Prefix for rotation parameters.
    phase_parameter_prefix : str, default "phi"
        Prefix for the phase‑shift parameters applied after each rotation block.
    name : str | None, default None
        Optional name for the resulting circuit.

    Returns
    -------
    QuantumCircuit
        A parameterized circuit implementing the extended ansatz.

    Notes
    -----
    - The circuit contains a total of ``num_rot_layers * num_qubits`` rotation parameters
      and ``num_rot_layers * num_qubits`` phase parameters, where
      ``num_rot_layers = reps`` if ``skip_final_rotation_layer`` else ``reps + 1``.
    - The rotation parameters are applied in alternating RX/RY fashion.
    - After each rotation block, a phase‑shift layer consisting of RZ gates is applied.
    - Entanglement is applied after each rotation block.
    """
    if num_qubits < 1:
        raise ValueError("num_qubits must be >= 1.")
    if reps < 0:
        raise ValueError("reps must be non‑negative.")

    n = int(num_qubits)
    qc = QuantumCircuit(n, name=name or "RealAmplitudesAlternatingExtended")

    num_rot_layers = reps if skip_final_rotation_layer else reps + 1

    # Parameter vectors
    rot_params = ParameterVector(parameter_prefix, num_rot_layers * n)
    phase_params = ParameterVector(phase_parameter_prefix, num_rot_layers * n)

    def _rot(layer: int) -> None:
        """Apply alternating RX/RY rotations for a given layer."""
        base = layer * n
        if layer % 2 == 0:
            for q in range(n):
                qc.ry(rot_params[base + q], q)
        else:
            for q in range(n):
                qc.rx(rot_params[base + q], q)

    def _phase_shift(layer: int) -> None:
        """Apply a phase‑shift (RZ) layer after rotations."""
        base = layer * n
        for q in range(n):
            qc.rz(phase_params[base + q], q)

    pairs = _resolve_entanglement(n, entanglement, max_entanglement_depth)

    for r in range(reps):
        _rot(r)
        _phase_shift(r)
        if insert_barriers:
            qc.barrier()
        for i, j in pairs:
            qc.cx(i, j)
        if insert_barriers:
            qc.barrier()

    if not skip_final_rotation_layer:
        _rot(reps)
        _phase_shift(reps)

    # Attach input parameters for easy binding
    qc.input_params = ParameterVector(parameter_prefix, num_rot_layers * n * 2)  # type: ignore[attr-defined]
    qc.input_params[:] = list(rot_params) + list(phase_params)  # type: ignore[attr-defined]
    qc.num_rot_layers = num_rot_layers  # type: ignore[attr-defined]
    return qc


class RealAmplitudesAlternatingExtended(QuantumCircuit):
    """Convenience subclass of QuantumCircuit for the extended ansatz."""

    def __init__(
        self,
        num_qubits: int,
        reps: int = 1,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Iterable[Tuple[int, int]]] = "full",
        max_entanglement_depth: int | None = None,
        skip_final_rotation_layer: bool = False,
        insert_barriers: bool = False,
        parameter_prefix: str = "theta",
        phase_parameter_prefix: str = "phi",
        name: str = "RealAmplitudesAlternatingExtended",
    ) -> None:
        built = real_amplitudes_alternating_extended(
            num_qubits,
            reps,
            entanglement,
            max_entanglement_depth,
            skip_final_rotation_layer,
            insert_barriers,
            parameter_prefix,
            phase_parameter_prefix,
            name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)
        self.input_params = built.input_params  # type: ignore[attr-defined]
        self.num_rot_layers = built.num_rot_layers  # type: ignore[attr-defined]
