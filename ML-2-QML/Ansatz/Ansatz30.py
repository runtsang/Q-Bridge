"""RealAmplitudes variant with extended expressivity.

This module builds upon the original alternating‑rotation
`real_amplitudes_alternating` ansatz and adds several new
features:

* **Rotation flexibility** – choose between RY, RX, RZ, or a
  custom sequence of rotation types per layer.
* **Parameter sharing** – share a single parameter across all qubits
  of a layer.
* **RZZ entanglement** – optional two‑qubit RZZ gates after each
  entanglement step, with dedicated parameters.
* **Custom entanglement schedules** – callable or sequence of
  pairs per layer, in addition to the built‑in “full”, “linear”,
  and “circular” options.
* **Barrier insertion** – optional barriers between layers for
  easier visual debugging.
* **Skip final entanglement** – optionally omit the last set of
  two‑qubit gates.

The public API mirrors the original: a convenience function
``real_amplitudes_alternating_extended`` and a
``RealAmplitudesAlternatingExtended`` subclass of :class:`~qiskit.QuantumCircuit`.

"""

from __future__ import annotations

from typing import Callable, Iterable, List, Sequence, Tuple, Union

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector


def _resolve_entanglement(
    num_qubits: int,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Iterable[Tuple[int, int]]],
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
    rotation_type: str | Sequence[str] = "alternating",
    parameter_sharing: bool = False,
    add_rzz_entanglement: bool = False,
    skip_final_entanglement_layer: bool = False,
    insert_barriers: bool = False,
    parameter_prefix: str = "theta",
    rzz_prefix: str = "phi",
    name: str | None = None,
) -> QuantumCircuit:
    """Construct a RealAmplitudes ansatz with flexible rotation and entanglement options.

    Args:
        num_qubits: Number of qubits in the circuit.
        reps: Number of rotation‑entanglement cycles.
        entanglement: Specification of two‑qubit pairs. Can be a string
            ("full", "linear", "circular") or a sequence of pairs.
        rotation_type: Either a single string ("alternating", "ry", "rx", "rz")
            or a sequence specifying the rotation type for each layer.
        parameter_sharing: If ``True``, use a single parameter per layer
            instead of one per qubit.
        add_rzz_entanglement: If ``True``, add an RZZ gate after each
            two‑qubit entanglement pair.
        skip_final_entanglement_layer: If ``True``, omit the final
            entanglement step after the last rotation layer.
        insert_barriers: Insert a barrier between layers for readability.
        parameter_prefix: Prefix for rotation parameters.
        rzz_prefix: Prefix for RZZ parameters (only used if
            ``add_rzz_entanglement`` is ``True``).
        name: Optional circuit name.

    Returns:
        A :class:`~qiskit.QuantumCircuit` instance representing the ansatz.
    """
    if num_qubits < 1:
        raise ValueError("num_qubits must be >= 1.")
    if reps < 1:
        raise ValueError("reps must be >= 1.")

    n = int(num_qubits)
    qc = QuantumCircuit(n, name=name or "RealAmplitudesExtended")

    # Determine rotation types per layer
    if isinstance(rotation_type, str):
        if rotation_type not in {"alternating", "ry", "rx", "rz"}:
            raise ValueError(f"Unsupported rotation_type string: {rotation_type!r}")
        rotation_seq = [rotation_type] * reps
    else:
        rotation_seq = list(rotation_type)
        if len(rotation_seq)!= reps:
            raise ValueError("Length of rotation_type sequence must equal reps.")

    # Parameter vectors
    if parameter_sharing:
        rot_params = ParameterVector(parameter_prefix, reps)
    else:
        rot_params = ParameterVector(parameter_prefix, reps * n)

    rzz_params: ParameterVector | None = None
    if add_rzz_entanglement:
        # One RZZ per pair per layer
        pairs = _resolve_entanglement(n, entanglement)
        rzz_params = ParameterVector(rzz_prefix, reps * len(pairs))

    def _apply_rotation(layer: int) -> None:
        """Apply a single rotation layer."""
        rot_type = rotation_seq[layer]
        if rot_type == "alternating":
            # Alternate RY and RX per qubit
            for q in range(n):
                angle = rot_params[layer * n + q] if not parameter_sharing else rot_params[layer]
                qc.ry(angle, q)
            for q in range(n):
                angle = rot_params[layer * n + q] if not parameter_sharing else rot_params[layer]
                qc.rx(angle, q)
        elif rot_type == "ry":
            for q in range(n):
                angle = rot_params[layer * n + q] if not parameter_sharing else rot_params[layer]
                qc.ry(angle, q)
        elif rot_type == "rx":
            for q in range(n):
                angle = rot_params[layer * n + q] if not parameter_sharing else rot_params[layer]
                qc.rx(angle, q)
        elif rot_type == "rz":
            for q in range(n):
                angle = rot_params[layer * n + q] if not parameter_sharing else rot_params[layer]
                qc.rz(angle, q)
        else:
            raise ValueError(f"Unsupported rotation type: {rot_type!r}")

    pairs = _resolve_entanglement(n, entanglement)

    for r in range(reps):
        _apply_rotation(r)
        if insert_barriers:
            qc.barrier()
        for idx, (i, j) in enumerate(pairs):
            qc.cx(i, j)
            if add_rzz_entanglement and rzz_params is not None:
                angle = rzz_params[r * len(pairs) + idx]
                qc.rzz(angle, i, j)
        if insert_barriers:
            qc.barrier()

    if not skip_final_entanglement_layer:
        for idx, (i, j) in enumerate(pairs):
            qc.cx(i, j)
            if add_rzz_entanglement and rzz_params is not None:
                angle = rzz_params[reps * len(pairs) + idx]
                qc.rzz(angle, i, j)

    # Attach metadata
    qc.input_params = rot_params  # type: ignore[attr-defined]
    qc.num_rot_layers = reps  # type: ignore[attr-defined]
    if add_rzz_entanglement:
        qc.rzz_params = rzz_params  # type: ignore[attr-defined]
    return qc


class RealAmplitudesAlternatingExtended(QuantumCircuit):
    """Convenience subclass wrapping :func:`real_amplitudes_alternating_extended`."""

    def __init__(
        self,
        num_qubits: int,
        reps: int = 1,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Iterable[Tuple[int, int]]] = "full",
        rotation_type: str | Sequence[str] = "alternating",
        parameter_sharing: bool = False,
        add_rzz_entanglement: bool = False,
        skip_final_entanglement_layer: bool = False,
        insert_barriers: bool = False,
        parameter_prefix: str = "theta",
        rzz_prefix: str = "phi",
        name: str = "RealAmplitudesExtended",
    ) -> None:
        built = real_amplitudes_alternating_extended(
            num_qubits,
            reps,
            entanglement,
            rotation_type,
            parameter_sharing,
            add_rzz_entanglement,
            skip_final_entanglement_layer,
            insert_barriers,
            parameter_prefix,
            rzz_prefix,
            name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)
        self.input_params = built.input_params  # type: ignore[attr-defined]
        self.num_rot_layers = built.num_rot_layers  # type: ignore[attr-defined]
        if add_rzz_entanglement:
            self.rzz_params = built.rzz_params  # type: ignore[attr-defined]


__all__ = [
    "RealAmplitudesAlternatingExtended",
    "real_amplitudes_alternating_extended",
]
