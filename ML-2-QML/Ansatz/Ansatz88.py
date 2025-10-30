"""RealAmplitudesCZExtended – a depth‑controlled, hybrid‑layer RealAmplitudes variant."""

from __future__ import annotations

from typing import Callable, List, Sequence, Tuple

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

__all__ = ["RealAmplitudesCZExtended", "real_amplitudes_cz_extended"]


# --------------------------------------------------------------------------- #
# Helper utilities
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
    entanglement : str/Sequence/Callable
        Entanglement definition.  Valid strings are
        * ``full`` – all pairs,
        * ``linear`` – single‑step linear chain,
        * ``circular`` – linear chain with the last qubit connected to the first.
        Any other string or a callable will be forwarded to the user‑defined routine.

    Returns
    -------
    list[Tuple[int, int]]
        List of two‑qubit pairs.

    Raises
    ------
    ValueError
        * If any pair is a single‑qubit or out‑of‑bounds.
        * If **entanglement** is a string that does not match a known pattern.
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
# Main ansatz function
# --------------------------------------------------------------------------- #
def real_amplitudes_cz_extended(
    num_qubits: int,
    reps: int = 1,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
    hybrid_depth: int = 0,
    skip_final_rotation_layer: bool = False,
    insert_barriers: bool = False,
    parameter_prefix: str = "theta",
    name: str | None = None,
) -> QuantumCircuit:
    """
    Construct a Real‑Amplitudes circuit with CZ entanglement and optional hybrid
    layers consisting of RX–RY rotations.

    The circuit is built as follows (for each repetition):

        1. Base rotation layer (single‑qubit Ry gates)
        2. Optional barrier
        3. CZ entanglement according to *entanglement*
        4. Optional barrier
        5. *hybrid_depth* hybrid rotation layers (RX followed by Ry per qubit)

    After all repetitions, a final base rotation layer is applied unless
    *skip_final_rotation_layer* is ``True``.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    reps : int, default 1
        Number of times the base+entanglement block is repeated.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]]
        Specification of the two‑qubit entanglement pattern.
    hybrid_depth : int, default 0
        Number of hybrid rotation layers per repetition (each uses RX and Ry).
    skip_final_rotation_layer : bool, default False
        If ``True`` the final base rotation layer after the last repetition is omitted.
    insert_barriers : bool, default False
        If ``True`` a barrier is inserted after each rotation and entanglement block.
    parameter_prefix : str, default "theta"
        Prefix for the parameters in the circuit.
    name : str | None, default None
        Name of the circuit.  If ``None`` a default name is used.

    Returns
    -------
    QuantumCircuit
        The constructed ansatz circuit.

    Notes
    -----
    * The total number of parameters is
      ``(reps + (0 if skip_final_rotation_layer else 1)) * num_qubits
      + reps * hybrid_depth * num_qubits * 2``.
    * Parameters are laid out in the circuit as follows:
      first all base‑layer parameters, then all hybrid‑layer parameters.
    """
    if num_qubits < 1:
        raise ValueError("num_qubits must be >= 1.")
    if reps < 1:
        raise ValueError("reps must be >= 1.")
    if hybrid_depth < 0:
        raise ValueError("hybrid_depth must be >= 0.")

    n = int(num_qubits)
    qc = QuantumCircuit(n, name=name or "RealAmplitudesCZExtended")

    base_layers = reps + (0 if skip_final_rotation_layer else 1)
    hybrid_layers = reps * hybrid_depth

    total_params = base_layers * n + hybrid_layers * n * 2
    params = ParameterVector(parameter_prefix, total_params)

    base_params = params[0 : base_layers * n]
    hybrid_params = params[base_layers * n :]

    def _rot(layer_idx: int) -> None:
        """Apply a single‑qubit Ry rotation layer."""
        base_slice = base_params[layer_idx * n : (layer_idx + 1) * n]
        for q in range(n):
            qc.ry(base_slice[q], q)

    def _hybrid_rot(layer_idx: int) -> None:
        """Apply an RX‑then‑RY hybrid rotation layer."""
        hybrid_slice = hybrid_params[layer_idx * n * 2 : (layer_idx + 1) * n * 2]
        for q in range(n):
            rx_param = hybrid_slice[2 * q]
            ry_param = hybrid_slice[2 * q + 1]
            qc.rx(rx_param, q)
            qc.ry(ry_param, q)

    pairs = _resolve_entanglement(n, entanglement)

    base_layer_idx = 0
    hybrid_layer_idx = 0

    for _ in range(reps):
        _rot(base_layer_idx)
        base_layer_idx += 1
        if insert_barriers:
            qc.barrier()
        for (i, j) in pairs:
            qc.cz(i, j)
        if insert_barriers:
            qc.barrier()
        for _ in range(hybrid_depth):
            _hybrid_rot(hybrid_layer_idx)
            hybrid_layer_idx += 1
            if insert_barriers:
                qc.barrier()

    if not skip_final_rotation_layer:
        _rot(base_layer_idx)

    qc.input_params = params
    qc.base_params = base_params
    qc.hybrid_params = hybrid_params
    qc.num_base_layers = base_layers
    qc.num_hybrid_layers = hybrid_layers
    qc.hybrid_depth = hybrid_depth
    return qc


# --------------------------------------------------------------------------- #
# Class wrapper
# --------------------------------------------------------------------------- #
class RealAmplitudesCZExtended(QuantumCircuit):
    """Class wrapper for the CZ‑entangling RealAmplitudes ansatz with hybrid layers.

    The class behaves like a standard ``QuantumCircuit``; it exposes the same
    convenience constructor as the functional API and stores the parameter
    vectors as attributes for easy inspection.
    """

    def __init__(
        self,
        num_qubits: int,
        reps: int = 1,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
        hybrid_depth: int = 0,
        skip_final_rotation_layer: bool = False,
        insert_barriers: bool = False,
        parameter_prefix: str = "theta",
        name: str = "RealAmplitudesCZExtended",
    ) -> None:
        built = real_amplitudes_cz_extended(
            num_qubits,
            reps,
            entanglement,
            hybrid_depth,
            skip_final_rotation_layer,
            insert_barriers,
            parameter_prefix,
            name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)
        self.input_params = built.input_params
        self.base_params = built.base_params
        self.hybrid_params = built.hybrid_params
        self.num_base_layers = built.num_base_layers
        self.num_hybrid_layers = built.num_hybrid_layers
        self.hybrid_depth = built.hybrid_depth
