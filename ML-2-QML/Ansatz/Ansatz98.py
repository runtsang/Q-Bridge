"""RealAmplitudes variant with alternating RY/RX rotation layers and controlled feature extensions."""
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


def real_amplitudes_alternating_controlled(
    num_qubits: int,
    reps: int = 1,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
    skip_final_rotation_layer: bool = False,
    insert_barriers: bool = False,
    parameter_prefix: str = "theta",
    parameter_sharing: bool = False,
    swap_layers: bool = False,
    name: str | None = None,
) -> QuantumCircuit:
    """
    Build a RealAmplitudes‑style ansatz with alternating RY / RX rotations.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    reps : int, default=1
        Number of alternating rotation + entanglement blocks (excluding the optional
        final rotation layer). Must be >= 1.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]]
        Specification of which qubit pairs to entangle. Accepts the same options as
        the original ansatz ("full", "linear", "circular") or a callable/sequence.
    skip_final_rotation_layer : bool, default=False
        If ``True`` the final rotation layer after the last entanglement block is omitted.
    insert_barriers : bool, default=False
        If ``True`` insert a barrier between each logical block for visual clarity.
    parameter_prefix : str, default="theta"
        Prefix for the rotation parameters.
    parameter_sharing : bool, default=False
        If ``True`` all rotation layers reuse a single set of parameters, reducing the
        total parameter count to ``num_qubits``. This imposes a symmetry across depth.
    swap_layers : bool, default=False
        If ``True`` the entanglement block is applied *before* the rotation block in
        each repetition. This changes the order of operations while keeping the same
        overall connectivity.
    name : str | None, default=None
        Optional name for the circuit. If ``None`` a default name is used.

    Returns
    -------
    QuantumCircuit
        The constructed ansatz circuit.

    Notes
    -----
    * Parameter sharing is a controlled modification that reduces circuit flexibility
      while still allowing the ansatz to capture global patterns.
    * Layer swapping is another controlled change that can be useful when the
      underlying hardware has stronger native support for entangling gates before
      single‑qubit rotations.
    """
    if num_qubits < 1:
        raise ValueError("num_qubits must be >= 1.")
    if reps < 1:
        raise ValueError("reps must be >= 1.")
    if not isinstance(parameter_sharing, bool):
        raise TypeError("parameter_sharing must be a bool.")
    if not isinstance(swap_layers, bool):
        raise TypeError("swap_layers must be a bool.")
    if not isinstance(insert_barriers, bool):
        raise TypeError("insert_barriers must be a bool.")
    if not isinstance(skip_final_rotation_layer, bool):
        raise TypeError("skip_final_rotation_layer must be a bool.")

    n = int(num_qubits)
    qc = QuantumCircuit(n, name=name or "RealAmplitudesAlternatingControlled")

    # Determine total rotation layers (include final layer if not skipped)
    num_rot_layers = reps if skip_final_rotation_layer else reps + 1

    # Parameter vector: shared across layers if requested
    param_len = n if parameter_sharing else num_rot_layers * n
    params = ParameterVector(parameter_prefix, param_len)

    def _rot(layer: int) -> None:
        """
        Apply the alternating rotation block for a given layer index.
        When parameter_sharing is enabled, the same parameters are reused across layers.
        """
        base = 0 if parameter_sharing else layer * n
        if layer % 2 == 0:
            for q in range(n):
                qc.ry(params[base + q], q)
        else:
            for q in range(n):
                qc.rx(params[base + q], q)

    pairs = _resolve_entanglement(n, entanglement)

    for r in range(reps):
        if swap_layers:
            # Entanglement first, then rotation
            for (i, j) in pairs:
                qc.cx(i, j)
            if insert_barriers:
                qc.barrier()
            _rot(r)
        else:
            # Original order: rotation then entanglement
            _rot(r)
            if insert_barriers:
                qc.barrier()
            for (i, j) in pairs:
                qc.cx(i, j)
        if insert_barriers:
            qc.barrier()

    # Final rotation layer (if not skipped)
    if not skip_final_rotation_layer:
        _rot(reps)

    qc.input_params = params  # type: ignore[attr-defined]
    qc.num_rot_layers = num_rot_layers  # type: ignore[attr-defined]
    return qc


class RealAmplitudesAlternatingControlled(QuantumCircuit):
    """Class wrapper for the controlled‑modification variant of RealAmplitudesAlternating."""

    def __init__(
        self,
        num_qubits: int,
        reps: int = 1,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
        skip_final_rotation_layer: bool = False,
        insert_barriers: bool = False,
        parameter_prefix: str = "theta",
        parameter_sharing: bool = False,
        swap_layers: bool = False,
        name: str = "RealAmplitudesAlternatingControlled",
    ) -> None:
        built = real_amplitudes_alternating_controlled(
            num_qubits,
            reps,
            entanglement,
            skip_final_rotation_layer,
            insert_barriers,
            parameter_prefix,
            parameter_sharing,
            swap_layers,
            name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)
        self.input_params = built.input_params  # type: ignore[attr-defined]
        self.num_rot_layers = built.num_rot_layers  # type: ignore[attr-defined]


__all__ = ["RealAmplitudesAlternatingControlled", "real_amplitudes_alternating_controlled"]
