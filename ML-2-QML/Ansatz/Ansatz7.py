"""RealAmplitudesCZShared: a parameter‑sharing variant of RealAmplitudesCZ."""
from __future__ import annotations

from typing import Callable, List, Sequence, Tuple

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector


# ----------------------------------------------------------------------
# Helper: resolve entanglement specification
# ----------------------------------------------------------------------
def _resolve_entanglement(
    num_qubits: int,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]],
) -> List[Tuple[int, int]]:
    """
    Return a list of two‑qubit entangling pairs.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]]
        Either a keyword describing a standard pattern (``"full"``, ``"linear"``,
        ``"circular"``) or an explicit list of pairs, or a callable that
        produces such a list given ``num_qubits``.

    Returns
    -------
    List[Tuple[int, int]]
        Validated list of distinct qubit pairs.

    Raises
    ------
    ValueError
        If an invalid string is supplied or a pair references an out‑of‑range
        qubit index.
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


# ----------------------------------------------------------------------
# Main ansatz builder
# ----------------------------------------------------------------------
def real_amplitudes_cz_shared(
    num_qubits: int,
    reps: int = 1,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
    skip_final_rotation_layer: bool = False,
    insert_barriers: bool = False,
    parameter_prefix: str = "theta",
    name: str | None = None,
    share_params: bool = True,
) -> QuantumCircuit:
    """
    Construct a RealAmplitudes circuit with CZ entanglement and optional
    parameter sharing across rotation layers.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the ansatz.
    reps : int, default 1
        Number of entanglement layers.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]]
        Specification of which qubit pairs receive CZ gates.
    skip_final_rotation_layer : bool, default False
        If True, omit the rotation layer after the last set of CZ gates.
    insert_barriers : bool, default False
        If True, insert a barrier between layers for easier circuit
        inspection and debugging.
    parameter_prefix : str, default "theta"
        Prefix for the automatically generated parameters.
    name : str | None, default None
        Name of the resulting QuantumCircuit.
    share_params : bool, default True
        If True, all rotation layers reuse the same ParameterVector.
        If False, each layer gets its own distinct parameters.

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
        raise ValueError("reps must be >= 0.")

    n = int(num_qubits)
    qc = QuantumCircuit(n, name=name or "RealAmplitudesCZShared")

    # Determine total number of rotation layers
    total_rot_layers = reps if skip_final_rotation_layer else reps + 1

    # Parameter handling
    if share_params:
        # Single ParameterVector reused for every layer
        params = ParameterVector(parameter_prefix, n)
    else:
        # One ParameterVector per rotation layer
        params = ParameterVector(parameter_prefix, total_rot_layers * n)

    def _rot(layer_index: int, param_offset: int = 0) -> None:
        """Apply a layer of Ry rotations."""
        base = param_offset if share_params else layer_index * n
        for q in range(n):
            qc.ry(params[base + q], q)

    pairs = _resolve_entanglement(n, entanglement)

    for r in range(reps):
        _rot(r)
        if insert_barriers:
            qc.barrier()
        for (i, j) in pairs:
            qc.cz(i, j)
        if insert_barriers:
            qc.barrier()

    if not skip_final_rotation_layer:
        _rot(reps)

    qc.input_params = params  # type: ignore[attr-defined]
    qc.num_rot_layers = total_rot_layers  # type: ignore[attr-defined]
    return qc


# ----------------------------------------------------------------------
# Class wrapper
# ----------------------------------------------------------------------
class RealAmplitudesCZShared(QuantumCircuit):
    """Convenience subclass for the parameter‑sharing variant of RealAmplitudesCZ."""

    def __init__(
        self,
        num_qubits: int,
        reps: int = 1,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
        skip_final_rotation_layer: bool = False,
        insert_barriers: bool = False,
        parameter_prefix: str = "theta",
        share_params: bool = True,
        name: str = "RealAmplitudesCZShared",
    ) -> None:
        built = real_amplitudes_cz_shared(
            num_qubits,
            reps,
            entanglement,
            skip_final_rotation_layer,
            insert_barriers,
            parameter_prefix,
            name,
            share_params,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)
        self.input_params = built.input_params  # type: ignore[attr-defined]
        self.num_rot_layers = built.num_rot_layers  # type: ignore[attr-defined]


__all__ = ["RealAmplitudesCZShared", "real_amplitudes_cz_shared"]
