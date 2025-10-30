"""RealAmplitudesCZExtended – a depth‑scaled, hybrid‑rotation ansatz for Qiskit.

The module provides two public interfaces:
* `real_amplitudes_cz_extended` – a function that builds and returns a QuantumCircuit.
* `RealAmplitudesCZExtended` – a subclass of QuantumCircuit that constructs the circuit in its
  constructor.

The ansatz extends the original RealAmplitudesCZ by:
  • a configurable depth multiplier that scales the number of repetitions;
  • a hybrid rotation block that can apply an RY followed optionally by an RZ on each qubit;
  • an optional final mixing rotation layer that can be toggled independently of the
    `skip_final_rotation_layer` flag.

Both interfaces expose the same keyword arguments, making the extended ansatz drop‑in
compatible with existing code that expects a RealAmplitudesCZ circuit.
"""

from __future__ import annotations

from typing import Callable, Iterable, List, Sequence, Tuple, Union

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

# --------------------------------------------------------------------------- #
# Helper types and functions
# --------------------------------------------------------------------------- #
EntanglementSpec = Union[
    str,
    Sequence[Tuple[int, int]],
    Callable[[int], Iterable[Tuple[int, int]]],
]


def _resolve_entanglement(
    num_qubits: int,
    entanglement: EntanglementSpec,
) -> List[Tuple[int, int]]:
    """
    Resolve the entanglement specification into a list of qubit pairs.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    entanglement : EntanglementSpec
        Either a string keyword, a sequence of (i, j) tuples,
        or a callable that returns such a sequence.

    Returns
    -------
    List[Tuple[int, int]]
        List of distinct qubit pairs.

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

    # Otherwise treat as a sequence
    pairs = [(int(i), int(j)) for (i, j) in entanglement]  # type: ignore[arg-type]
    for i, j in pairs:
        if i == j:
            raise ValueError("Entanglement pairs must connect distinct qubits.")
        if not (0 <= i < num_qubits and 0 <= j < num_qubits):
            raise ValueError(
                f"Entanglement pair {(i, j)} out of range for n={num_qubits}."
            )
    return pairs


def _apply_rotation_layer(
    qc: QuantumCircuit,
    params: ParameterVector,
    start_idx: int,
    num_qubits: int,
    use_rz: bool,
) -> int:
    """
    Apply a single‑qubit rotation block to all qubits.

    Parameters
    ----------
    qc : QuantumCircuit
        Target circuit.
    params : ParameterVector
        Vector of parameters for all rotation layers.
    start_idx : int
        Index of the first parameter for this layer.
    num_qubits : int
        Number of qubits.
    use_rz : bool
        If True, add an RZ after each RY.

    Returns
    -------
    int
        Number of parameters consumed by this layer.
    """
    # RY for each qubit
    for q in range(num_qubits):
        qc.ry(params[start_idx + q], q)

    consumed = num_qubits
    if use_rz:
        # RZ for each qubit
        for q in range(num_qubits):
            qc.rz(params[start_idx + consumed + q], q)
        consumed += num_qubits
    return consumed


def _apply_entanglement_block(
    qc: QuantumCircuit,
    pairs: List[Tuple[int, int]],
) -> None:
    """
    Apply CZ gates for each pair in the entanglement schedule.

    Parameters
    ----------
    qc : QuantumCircuit
        Target circuit.
    pairs : List[Tuple[int, int]]
        List of qubit pairs to entangle.
    """
    for i, j in pairs:
        qc.cz(i, j)


# --------------------------------------------------------------------------- #
# Public API – function
# --------------------------------------------------------------------------- #
def real_amplitudes_cz_extended(
    num_qubits: int,
    reps: int = 1,
    entanglement: EntanglementSpec = "full",
    skip_final_rotation_layer: bool = False,
    insert_barriers: bool = False,
    parameter_prefix: str = "theta",
    name: str | None = None,
    depth_multiplier: int = 1,
    use_rz: bool = False,
    mixing: bool = False,
) -> QuantumCircuit:
    """
    Construct a depth‑scaled, hybrid‑rotation RealAmplitudesCZ circuit.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    reps : int, default 1
        Number of base repetitions (before depth scaling).
    entanglement : EntanglementSpec, default "full"
        Entanglement schedule.
    skip_final_rotation_layer : bool, default False
        If True, the final rotation layer is omitted.
    insert_barriers : bool, default False
        If True, insert barriers between logical layers.
    parameter_prefix : str, default "theta"
        Prefix for the generated parameters.
    name : str, default None
        Name for the circuit; if None the default name is used.
    depth_multiplier : int, default 1
        Factor by which the number of repetitions is scaled.
    use_rz : bool, default False
        If True, each rotation layer includes an RZ after the RY.
    mixing : bool, default False
        If True, append an additional mixing rotation layer after the final entanglement.

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
    if reps < 1:
        raise ValueError("reps must be >= 1.")
    if depth_multiplier < 1:
        raise ValueError("depth_multiplier must be >= 1.")

    total_reps = reps * depth_multiplier
    n = int(num_qubits)

    qc = QuantumCircuit(n, name=name or "RealAmplitudesCZExtended")

    # Resolve entanglement pairs
    pairs = _resolve_entanglement(n, entanglement)

    # Compute total number of rotation layers
    rot_layers_per_rep = 1  # base rotation layer per repetition
    final_layer = 0 if skip_final_rotation_layer else 1
    mixing_layer = 1 if mixing else 0
    total_rot_layers = total_reps * rot_layers_per_rep + final_layer + mixing_layer

    # Each rotation layer may contain RY (and optionally RZ)
    params_per_layer = n * (1 + int(use_rz))
    total_params = total_rot_layers * params_per_layer

    params = ParameterVector(parameter_prefix, total_params)

    # Index into the parameter vector
    param_ptr = 0

    for _ in range(total_reps):
        # Rotation layer
        consumed = _apply_rotation_layer(qc, params, param_ptr, n, use_rz)
        param_ptr += consumed

        if insert_barriers:
            qc.barrier()

        # Entanglement block
        _apply_entanglement_block(qc, pairs)

        if insert_barriers:
            qc.barrier()

    # Final rotation layer (if not skipped)
    if not skip_final_rotation_layer:
        consumed = _apply_rotation_layer(qc, params, param_ptr, n, use_rz)
        param_ptr += consumed

    # Optional mixing layer
    if mixing:
        consumed = _apply_rotation_layer(qc, params, param_ptr, n, use_rz)
        param_ptr += consumed

    qc.input_params = params  # type: ignore[attr-defined]
    qc.num_rot_layers = total_rot_layers  # type: ignore[attr-defined]
    return qc


# --------------------------------------------------------------------------- #
# Public API – class
# --------------------------------------------------------------------------- #
class RealAmplitudesCZExtended(QuantumCircuit):
    """
    Subclass of QuantumCircuit implementing the extended RealAmplitudesCZ ansatz.

    The constructor accepts the same keyword arguments as
    :func:`real_amplitudes_cz_extended`.  The circuit is built and composed
    during initialization, making it ready for immediate use in Qiskit workflows.
    """

    def __init__(
        self,
        num_qubits: int,
        reps: int = 1,
        entanglement: EntanglementSpec = "full",
        skip_final_rotation_layer: bool = False,
        insert_barriers: bool = False,
        parameter_prefix: str = "theta",
        name: str = "RealAmplitudesCZExtended",
        depth_multiplier: int = 1,
        use_rz: bool = False,
        mixing: bool = False,
    ) -> None:
        built = real_amplitudes_cz_extended(
            num_qubits,
            reps,
            entanglement,
            skip_final_rotation_layer,
            insert_barriers,
            parameter_prefix,
            name,
            depth_multiplier,
            use_rz,
            mixing,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)
        self.input_params = built.input_params  # type: ignore[attr-defined]
        self.num_rot_layers = built.num_rot_layers  # type: ignore[attr-defined]


__all__ = ["RealAmplitudesCZExtended", "real_amplitudes_cz_extended"]
