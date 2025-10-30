"""RealAmplitudes variant with alternating RY/RX rotation layers and extended entanglement."""
from __future__ import annotations

from typing import Callable, Iterable, List, Sequence, Tuple

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

__all__ = [
    "RealAmplitudesAlternatingExtended",
    "real_amplitudes_alternating_extended",
]


def _resolve_entanglement(
    num_qubits: int,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Iterable[Tuple[int, int]]],
) -> List[Tuple[int, int]]:
    """Return a list of two‑qubit pairs based on a simple entanglement spec.

    The function accepts a string, a sequence, or a callable returning an iterable of pairs.
    It performs bounds‑check and raises informative error messages for invalid pairs.
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


def real_amplitudes_alternating_extended(
    num_qubits: int,
    reps: int = 1,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Iterable[Tuple[int, int]]] = "full",
    depth_schedule: Sequence[int] | None = None,
    entanglement_schedule: Sequence[
        str | Sequence[Tuple[int, int]] | Callable[[int], Iterable[Tuple[int, int]]]
    ] | None = None,
    mid_entanglement: bool = False,
    skip_final_rotation_layer: bool = False,
    insert_barriers: bool = False,
    parameter_prefix: str = "theta",
    name: str | None = None,
) -> QuantumCircuit:
    """Return a parameterized circuit with alternating RY/RX layers and configurable entanglement.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the ansatz.
    reps : int, default 1
        Number of repetition blocks.
    entanglement : str or sequence or callable, default "full"
        Entanglement pattern to use when ``entanglement_schedule`` is None.
    depth_schedule : Sequence[int] or None, default None
        Number of rotation layers per repetition.  If ``None`` defaults to ``[1]*reps``.
    entanglement_schedule : Sequence[spec] or None, default None
        Sequence of entanglement specifications—one per repetition.  If ``None`` defaults to
        ``[entanglement]*reps``.
    mid_entanglement : bool, default False
        If True, apply an entanglement layer between every rotation layer within a repetition.
    skip_final_rotation_layer : bool, default False
        Retained for API compatibility; ignored in the extended design.
    insert_barriers : bool, default False
        Insert barriers between logical blocks for clearer visualisation.
    parameter_prefix : str, default "theta"
        Prefix for the parameter names.
    name : str or None, default None
        Circuit name; defaults to ``"RealAmplitudesAlternatingExtended"``.

    Returns
    -------
    QuantumCircuit
        Parameterised ansatz circuit.

    Notes
    -----
    * The rotation layers alternate between RY and RX globally: even‑indexed layers use RY, odd
      layers use RX.
    * The total number of rotation layers is ``sum(depth_schedule)``.
    * The ``depth_schedule`` and ``entanglement_schedule`` must both have length ``reps``.
    """
    if num_qubits < 1:
        raise ValueError("num_qubits must be >= 1.")
    if reps < 1:
        raise ValueError("reps must be >= 1.")
    n = int(num_qubits)

    # Resolve depth schedule
    if depth_schedule is None:
        depth_schedule = [1] * reps
    if len(depth_schedule)!= reps:
        raise ValueError("depth_schedule must have length equal to reps.")
    # Resolve entanglement schedule
    if entanglement_schedule is None:
        entanglement_schedule = [entanglement] * reps
    if len(entanglement_schedule)!= reps:
        raise ValueError("entanglement_schedule must have length equal to reps.")

    total_rot_layers = sum(depth_schedule)
    params = ParameterVector(parameter_prefix, total_rot_layers * n)

    qc = QuantumCircuit(n, name=name or "RealAmplitudesAlternatingExtended")

    def _rot(layer: int) -> None:
        base = layer * n
        if layer % 2 == 0:
            for q in range(n):
                qc.ry(params[base + q], q)
        else:
            for q in range(n):
                qc.rx(params[base + q], q)

    layer_counter = 0
    for r in range(reps):
        # rotation layers for this repetition
        for _ in range(depth_schedule[r]):
            _rot(layer_counter)
            layer_counter += 1
            if insert_barriers:
                qc.barrier()
            if mid_entanglement:
                pairs = _resolve_entanglement(n, entanglement_schedule[r])
                for (i, j) in pairs:
                    qc.cx(i, j)
                if insert_barriers:
                    qc.barrier()

        # final entanglement for this repetition
        pairs = _resolve_entanglement(n, entanglement_schedule[r])
        for (i, j) in pairs:
            qc.cx(i, j)
        if insert_barriers:
            qc.barrier()

    qc.input_params = params  # type: ignore[attr-defined]
    qc.num_rot_layers = total_rot_layers  # type: ignore[attr-defined]
    return qc


class RealAmplitudesAlternatingExtended(QuantumCircuit):
    """Convenience subclass exposing the extended alternating ansatz as a circuit."""

    def __init__(
        self,
        num_qubits: int,
        reps: int = 1,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Iterable[Tuple[int, int]]] = "full",
        depth_schedule: Sequence[int] | None = None,
        entanglement_schedule: Sequence[
            str | Sequence[Tuple[int, int]] | Callable[[int], Iterable[Tuple[int, int]]]
        ] | None = None,
        mid_entanglement: bool = False,
        skip_final_rotation_layer: bool = False,
        insert_barriers: bool = False,
        parameter_prefix: str = "theta",
        name: str = "RealAmplitudesAlternatingExtended",
    ) -> None:
        built = real_amplitudes_alternating_extended(
            num_qubits,
            reps,
            entanglement,
            depth_schedule,
            entanglement_schedule,
            mid_entanglement,
            skip_final_rotation_layer,
            insert_barriers,
            parameter_prefix,
            name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)
        self.input_params = built.input_params  # type: ignore[attr-defined]
        self.num_rot_layers = built.num_rot_layers  # type: ignore[attr-defined]
