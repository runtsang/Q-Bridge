"""
RealAmplitudesCZExtended: a depth‑controlled, hybrid ansatz that extends the original RealAmplitudesCZ by adding
global‑phase blocks and a flexible depth scaling mechanism.

Typical usage:

>>> from qiskit import QuantumCircuit
>>> qc = RealAmplitudesCZExtended(num_qubits=3, depth=2, use_global_phase=True)
>>> qc.draw('mpl')

The design keeps the same quantum circuit structure but augments it with:
- A *depth* parameter that controls how many times the rotation–entanglement blocks are repeated.
- A *use_global_phase* flag that inserts a controlled‑phase (RZ) block after each entanglement pair.
- Optional *insert_barriers* to aid visual debugging.
"""

from __future__ import annotations

from typing import Callable, List, Sequence, Tuple

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector


def _resolve_entanglement(
    num_qubits: int,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]],
) -> List[Tuple[int, int]]:
    """
    Translate a simple entanglement specification into a list of two‑qubit pairs.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]]
        Either a string describing a predefined pattern, a concrete list of pairs, or a
        callable that generates the pairs given the qubit count.

    Returns
    -------
    List[Tuple[int, int]]
        Ordered list of distinct qubit pairs.

    Raises
    ------
    ValueError
        If the specification is invalid or contains out‑of‑range indices.
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


def real_amplitudes_cz_extended(
    num_qubits: int,
    depth: int = 1,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
    skip_final_rotation_layer: bool = False,
    insert_barriers: bool = False,
    use_global_phase: bool = False,
    parameter_prefix: str = "theta",
    phase_prefix: str = "phi",
    name: str | None = None,
) -> QuantumCircuit:
    """
    Build a depth‑controlled RealAmplitudes CZ ansatz with optional global‑phase layers.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    depth : int, default 1
        Number of rotation–entanglement cycles.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]], default "full"
        Specification of the entanglement pattern.
    skip_final_rotation_layer : bool, default False
        If True, omit the final rotation layer after the last entanglement.
    insert_barriers : bool, default False
        If True, insert barriers between logical blocks for easier visual inspection.
    use_global_phase : bool, default False
        If True, insert an RZ (global‑phase) layer after each entanglement.
    parameter_prefix : str, default "theta"
        Prefix for rotation parameters.
    phase_prefix : str, default "phi"
        Prefix for phase parameters when `use_global_phase` is True.
    name : str | None, default None
        Name of the resulting QuantumCircuit.

    Returns
    -------
    QuantumCircuit
        The constructed ansatz circuit.

    Raises
    ------
    ValueError
        If any argument is invalid.
    """
    if num_qubits < 1:
        raise ValueError("num_qubits must be >= 1.")
    if depth < 1:
        raise ValueError("depth must be >= 1.")
    if not isinstance(skip_final_rotation_layer, bool):
        raise ValueError("skip_final_rotation_layer must be a bool.")
    if not isinstance(insert_barriers, bool):
        raise ValueError("insert_barriers must be a bool.")
    if not isinstance(use_global_phase, bool):
        raise ValueError("use_global_phase must be a bool.")
    if not isinstance(parameter_prefix, str):
        raise ValueError("parameter_prefix must be a string.")
    if not isinstance(phase_prefix, str):
        raise ValueError("phase_prefix must be a string.")

    n = int(num_qubits)
    qc = QuantumCircuit(n, name=name or "RealAmplitudesCZExtended")

    # Rotation layers
    num_rot_layers = depth if skip_final_rotation_layer else depth + 1
    rot_params = ParameterVector(parameter_prefix, num_rot_layers * n)

    # Phase layers (optional)
    phase_params = None
    if use_global_phase:
        phase_params = ParameterVector(phase_prefix, depth * n)

    def _rot(layer: int) -> None:
        base = layer * n
        for q in range(n):
            qc.ry(rot_params[base + q], q)

    def _phase(layer: int) -> None:
        if not use_global_phase:
            return
        base = layer * n
        for q in range(n):
            qc.rz(phase_params[base + q], q)

    pairs = _resolve_entanglement(n, entanglement)

    for r in range(depth):
        _rot(r)
        if insert_barriers:
            qc.barrier()
        for (i, j) in pairs:
            qc.cz(i, j)
        if insert_barriers:
            qc.barrier()
        _phase(r)
        if insert_barriers:
            qc.barrier()

    if not skip_final_rotation_layer:
        _rot(depth)
        if insert_barriers:
            qc.barrier()

    qc.input_params = rot_params  # type: ignore[attr-defined]
    if use_global_phase:
        qc.phase_params = phase_params  # type: ignore[attr-defined]
    qc.num_rot_layers = num_rot_layers  # type: ignore[attr-defined]
    qc.num_phase_layers = depth if use_global_phase else 0  # type: ignore[attr-defined]
    qc.num_entanglement_layers = depth  # type: ignore[attr-defined]
    return qc


class RealAmplitudesCZExtended(QuantumCircuit):
    """Convenience wrapper that builds a RealAmplitudesCZExtended ansatz as a QuantumCircuit subclass."""

    def __init__(
        self,
        num_qubits: int,
        depth: int = 1,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
        skip_final_rotation_layer: bool = False,
        insert_barriers: bool = False,
        use_global_phase: bool = False,
        parameter_prefix: str = "theta",
        phase_prefix: str = "phi",
        name: str = "RealAmplitudesCZExtended",
    ) -> None:
        built = real_amplitudes_cz_extended(
            num_qubits,
            depth,
            entanglement,
            skip_final_rotation_layer,
            insert_barriers,
            use_global_phase,
            parameter_prefix,
            phase_prefix,
            name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)
        self.input_params = built.input_params  # type: ignore[attr-defined]
        self.phase_params = getattr(built, "phase_params", None)  # type: ignore[attr-defined]
        self.num_rot_layers = built.num_rot_layers  # type: ignore[attr-defined]
        self.num_phase_layers = built.num_phase_layers  # type: ignore[attr-defined]
        self.num_entanglement_layers = built.num_entanglement_layers  # type: ignore[attr-defined]


__all__ = ["RealAmplitudesCZExtended", "real_amplitudes_cz_extended"]
