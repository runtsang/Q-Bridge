"""
RealAmplitudesAlternatingExtended
A Qiskit-compatible ansatz that extends the alternating-rotation RealAmplitudes variant.
The extension introduces:
  * a configurable hybrid depth (multiple rotation‑entanglement cycles per repetition),
  * an optional controlled‑Z entanglement block,
  * barrier placement between logical blocks,
  * flexible entanglement schedule.
The rotation pattern remains alternating across all layers.
"""

from __future__ import annotations

from typing import Callable, List, Sequence, Tuple

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

# --------------------------------------------------------------------------- #
# Helper: Resolve entanglement specification
# --------------------------------------------------------------------------- #
def _resolve_entanglement(
    num_qubits: int,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]],
) -> List[Tuple[int, int]]:
    """Return a list of two‑qubit pairs according to a simple entanglement spec.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]]
        Entanglement specification:
        * ``"full"``   : all-to-all pairs.
        * ``"linear"`` : nearest‑neighbour linear chain.
        * ``"circular"`` : linear chain with an extra connection from the last to the first.
        * A list of (i, j) tuples.
        * A callable that accepts ``num_qubits`` and returns a sequence of tuples.

    Returns
    -------
    List[Tuple[int, int]]
        Validated list of qubit pairs.

    Raises
    ------
    ValueError
        If an invalid specification is supplied or a pair references the same qubit.
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

# --------------------------------------------------------------------------- #
# Ansatz construction
# --------------------------------------------------------------------------- #
def real_amplitudes_alternating_extended(
    num_qubits: int,
    reps: int = 1,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
    skip_final_rotation_layer: bool = False,
    insert_barriers: bool = False,
    use_controlled_z: bool = False,
    hybrid_depth: int = 1,
    parameter_prefix: str = "theta",
    name: str | None = None,
) -> QuantumCircuit:
    """Create a RealAmplitudes‑style ansatz with alternating RX/RY rotations and a hybrid depth.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    reps : int, optional
        Number of repetition blocks.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]], optional
        Entanglement schedule.
    skip_final_rotation_layer : bool, optional
        If ``True`` the final rotation layer after the last repetition is omitted.
    insert_barriers : bool, optional
        If ``True`` a barrier is inserted after each logical block.
    use_controlled_z : bool, optional
        If ``True`` a controlled‑Z gate is applied after each CX entanglement block.
    hybrid_depth : int, optional
        Number of rotation‑entanglement cycles per repetition. Must be >= 1.
    parameter_prefix : str, optional
        Prefix for the parameter names.
    name : str | None, optional
        Name of the circuit.

    Returns
    -------
    QuantumCircuit
        The constructed ansatz.

    Raises
    ------
    ValueError
        If ``num_qubits`` is less than 1 or ``hybrid_depth`` is not positive.
    """
    if num_qubits < 1:
        raise ValueError("num_qubits must be >= 1.")
    if hybrid_depth < 1:
        raise ValueError("hybrid_depth must be >= 1.")

    n = int(num_qubits)
    qc = QuantumCircuit(n, name=name or "RealAmplitudesAlternatingExtended")

    # Total number of rotation layers across the whole circuit
    total_rot_layers = reps * hybrid_depth + (0 if skip_final_rotation_layer else 1)
    params = ParameterVector(parameter_prefix, total_rot_layers * n)

    # Helper to apply a single rotation layer
    def _rot(layer_idx: int) -> None:
        base = layer_idx * n
        if layer_idx % 2 == 0:
            for q in range(n):
                qc.ry(params[base + q], q)
        else:
            for q in range(n):
                qc.rx(params[base + q], q)

    pairs = _resolve_entanglement(n, entanglement)

    layer_counter = 0  # global rotation layer index

    for r in range(reps):
        for d in range(hybrid_depth):
            _rot(layer_counter)
            layer_counter += 1
            if insert_barriers:
                qc.barrier()
            for (i, j) in pairs:
                qc.cx(i, j)
            if use_controlled_z:
                for (i, j) in pairs:
                    qc.cz(i, j)
            if insert_barriers:
                qc.barrier()

    if not skip_final_rotation_layer:
        _rot(layer_counter)

    qc.input_params = params  # type: ignore[attr-defined]
    qc.num_rot_layers = total_rot_layers  # type: ignore[attr-defined]
    return qc

# --------------------------------------------------------------------------- #
# Class wrapper
# --------------------------------------------------------------------------- #
class RealAmplitudesAlternatingExtended(QuantumCircuit):
    """Convenience subclass that builds the extended ansatz.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    reps : int, optional
        Number of repetition blocks.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]], optional
        Entanglement schedule.
    skip_final_rotation_layer : bool, optional
        If ``True`` the final rotation layer after the last repetition is omitted.
    insert_barriers : bool, optional
        If ``True`` a barrier is inserted after each logical block.
    use_controlled_z : bool, optional
        If ``True`` a controlled‑Z gate is applied after each CX entanglement block.
    hybrid_depth : int, optional
        Number of rotation‑entanglement cycles per repetition. Must be >= 1.
    parameter_prefix : str, optional
        Prefix for the parameter names.
    name : str, optional
        Name of the circuit.

    Attributes
    ----------
    input_params : ParameterVector
        Vector of symbolic parameters.
    num_rot_layers : int
        Total number of rotation layers.
    """

    def __init__(
        self,
        num_qubits: int,
        reps: int = 1,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
        skip_final_rotation_layer: bool = False,
        insert_barriers: bool = False,
        use_controlled_z: bool = False,
        hybrid_depth: int = 1,
        parameter_prefix: str = "theta",
        name: str = "RealAmplitudesAlternatingExtended",
    ) -> None:
        built = real_amplitudes_alternating_extended(
            num_qubits,
            reps,
            entanglement,
            skip_final_rotation_layer,
            insert_barriers,
            use_controlled_z,
            hybrid_depth,
            parameter_prefix,
            name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)
        self.input_params = built.input_params  # type: ignore[attr-defined]
        self.num_rot_layers = built.num_rot_layers  # type: ignore[attr-defined]

__all__ = [
    "RealAmplitudesAlternatingExtended",
    "real_amplitudes_alternating_extended",
]
