"""RealAmplitudes Alternating with optional reflection symmetry."""
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


def real_amplitudes_alternating_symmetry(
    num_qubits: int,
    reps: int = 1,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
    skip_final_rotation_layer: bool = False,
    insert_barriers: bool = False,
    parameter_prefix: str = "theta",
    symmetry: bool = True,
    name: str | None = None,
) -> QuantumCircuit:
    """
    Construct a Qiskit QuantumCircuit implementing an alternating‑rotation ansatz
    with optional reflection symmetry.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit. Must be >= 1.
    reps : int, default 1
        Number of repetition blocks of rotation + entanglement.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]]
        Specification of the two‑qubit entanglement pairs.  Accepted strings are
        ``"full"``, ``"linear"``, and ``"circular"``.  Alternatively a sequence
        of index pairs or a callable that returns such a sequence can be provided.
    skip_final_rotation_layer : bool, default False
        If ``True`` the final rotation layer after the last entanglement block
        is omitted, mirroring the behaviour of the original seed.
    insert_barriers : bool, default False
        If ``True`` a barrier is inserted between each rotation and entanglement
        block to aid classical optimisers.
    parameter_prefix : str, default "theta"
        Prefix for the parameter names.
    symmetry : bool, default True
        When ``True`` rotation parameters are shared across mirror‑symmetric qubits
        (q and n-1-q).  When ``False`` the circuit reverts to the original fully
        parameterised behaviour.
    name : str | None, default None
        Optional name for the circuit.  If ``None`` a default is used.

    Returns
    -------
    QuantumCircuit
        The constructed ansatz circuit.  The circuit exposes two attributes:
        ``input_params`` (the ParameterVector) and ``num_rot_layers`` (int).

    Raises
    ------
    ValueError
        If ``num_qubits`` < 1 or ``reps`` < 0.
    """
    if num_qubits < 1:
        raise ValueError("num_qubits must be >= 1.")
    if reps < 0:
        raise ValueError("reps must be non‑negative.")

    n = int(num_qubits)
    qc = QuantumCircuit(n, name=name or "RealAmplitudesAlternatingSymmetry")

    # Number of rotation layers: one per repetition plus an optional final layer.
    num_rot_layers = reps if skip_final_rotation_layer else reps + 1

    # Parameter vector: either full or symmetry‑shared.
    if symmetry:
        unique_qubits = (n + 1) // 2  # mirror pairs + middle qubit if odd
        param_count = num_rot_layers * unique_qubits
    else:
        param_count = num_rot_layers * n
    params = ParameterVector(parameter_prefix, param_count)

    def _rot(layer: int) -> None:
        """Apply the alternating rotation layer."""
        base = layer * n
        rot_func = qc.ry if layer % 2 == 0 else qc.rx
        for q in range(n):
            if symmetry:
                idx = base + min(q, n - 1 - q)
            else:
                idx = base + q
            rot_func(params[idx], q)

    pairs = _resolve_entanglement(n, entanglement)

    for r in range(reps):
        _rot(r)
        if insert_barriers:
            qc.barrier()
        for (i, j) in pairs:
            qc.cx(i, j)
        if insert_barriers:
            qc.barrier()

    if not skip_final_rotation_layer:
        _rot(reps)

    qc.input_params = params  # type: ignore[attr-defined]
    qc.num_rot_layers = num_rot_layers  # type: ignore[attr-defined]
    return qc


class RealAmplitudesAlternatingSymmetry(QuantumCircuit):
    """Convenience subclass for the symmetry‑aware alternating‑rotation ansatz."""

    def __init__(
        self,
        num_qubits: int,
        reps: int = 1,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
        skip_final_rotation_layer: bool = False,
        insert_barriers: bool = False,
        parameter_prefix: str = "theta",
        symmetry: bool = True,
        name: str = "RealAmplitudesAlternatingSymmetry",
    ) -> None:
        built = real_amplitudes_alternating_symmetry(
            num_qubits,
            reps,
            entanglement,
            skip_final_rotation_layer,
            insert_barriers,
            parameter_prefix,
            symmetry,
            name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)
        self.input_params = built.input_params  # type: ignore[attr-defined]
        self.num_rot_layers = built.num_rot_layers  # type: ignore[attr-defined]


__all__ = [
    "RealAmplitudesAlternatingSymmetry",
    "real_amplitudes_alternating_symmetry",
]
