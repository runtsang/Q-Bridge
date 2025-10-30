"""RealAmplitudes variant with global Z‑parity symmetry (parameter‑sharing)."""

from __future__ import annotations

from typing import Callable, List, Sequence, Tuple

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

__all__ = [
    "RealAmplitudesCZSymmetric",
    "real_amplitudes_cz_symmetric",
]


def _resolve_entanglement(
    num_qubits: int,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]],
) -> List[Tuple[int, int]]:
    """
    Resolve entanglement specification into a list of two‑qubit pairs.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]]
        Entanglement specification. Supported strings are ``"full"``, ``"linear"``, and ``"circular"``.
        Alternatively a sequence of pairs or a callable returning such a sequence.

    Returns
    -------
    List[Tuple[int, int]]
        Validated list of qubit pairs for entanglement.

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


def real_amplitudes_cz_symmetric(
    num_qubits: int,
    reps: int = 1,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
    skip_final_rotation_layer: bool = False,
    insert_barriers: bool = False,
    parameter_prefix: str = "theta",
    name: str | None = None,
) -> QuantumCircuit:
    """
    Construct a RealAmplitudes circuit with CZ entanglers and global Z‑parity symmetry.

    The ansatz mirrors rotation parameters across symmetric qubit pairs, reducing the
    number of independent parameters to ceil(n/2) per rotation layer.  The entanglement
    pattern and optional barriers are identical to the original RealAmplitudesCZ.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    reps : int, default 1
        Number of repetition layers of rotation + entanglement.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]], default "full"
        Entanglement specification.
    skip_final_rotation_layer : bool, default False
        If True, omit the final rotation layer after the last entanglement.
    insert_barriers : bool, default False
        If True, insert barriers between rotation and entanglement sub‑layers.
    parameter_prefix : str, default "theta"
        Prefix for parameter vector names.
    name : str | None, default None
        Optional circuit name.

    Returns
    -------
    QuantumCircuit
        Parameterised circuit with CZ entanglers and symmetric rotations.

    Raises
    ------
    ValueError
        If ``num_qubits`` is less than 1 or if ``reps`` is negative.
    """
    if num_qubits < 1:
        raise ValueError("num_qubits must be >= 1.")
    if reps < 0:
        raise ValueError("reps must be non‑negative.")

    n = int(num_qubits)
    qc = QuantumCircuit(n, name=name or "RealAmplitudesCZSymmetric")

    num_rot_layers = reps if skip_final_rotation_layer else reps + 1
    half_qubits = (n + 1) // 2  # number of independent parameters per layer
    params = ParameterVector(parameter_prefix, num_rot_layers * half_qubits)

    def _rot(layer: int) -> None:
        """Apply a rotation layer with parameter mirroring."""
        base = layer * half_qubits
        for i in range(half_qubits):
            param = params[base + i]
            qc.ry(param, i)
            # Mirror onto the symmetric qubit if it is distinct
            sym = n - i - 1
            if sym!= i:
                qc.ry(param, sym)

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
    qc.num_rot_layers = num_rot_layers  # type: ignore[attr-defined]
    return qc


class RealAmplitudesCZSymmetric(QuantumCircuit):
    """Class wrapper for the symmetric RealAmplitudes CZ ansatz."""

    def __init__(
        self,
        num_qubits: int,
        reps: int = 1,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
        skip_final_rotation_layer: bool = False,
        insert_barriers: bool = False,
        parameter_prefix: str = "theta",
        name: str = "RealAmplitudesCZSymmetric",
    ) -> None:
        built = real_amplitudes_cz_symmetric(
            num_qubits,
            reps,
            entanglement,
            skip_final_rotation_layer,
            insert_barriers,
            parameter_prefix,
            name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)
        self.input_params = built.input_params  # type: ignore[attr-defined]
        self.num_rot_layers = built.num_rot_layers  # type: ignore[attr-defined]
