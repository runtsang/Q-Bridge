"""RealAmplitudesCZExtended: a depth‑controlled hybrid ansatz that builds upon the original RealAmplitudesCZ.

The design adds:
* A configurable ``max_depth`` that allows the **depth** (num_reps + 1) to be scaled up to any integer.
* **Hybrid** – one‑qubit rotation‑only layers are automatically generated before and after each CZ‑entanglement block.
* Optional SWAP networks to enhance connectivity.
* Flexible entanglement schedules that can be static strings, explicit pair lists, or callables.
"""

from __future__ import annotations

from typing import Callable, List, Sequence, Tuple

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector


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
        Entanglement specification. Supported strings are ``'full'``, ``'linear'`` and ``'circular'``.
        A list of pairs or a callable returning such a list is also accepted.

    Returns
    -------
    List[Tuple[int, int]]
        Validated list of qubit pairs.
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


def _apply_swaps(qc: QuantumCircuit, pairs: Sequence[Tuple[int, int]]) -> None:
    """Apply a SWAP network to the circuit.

    Parameters
    ----------
    qc : QuantumCircuit
        The circuit to modify.
    pairs : Sequence[Tuple[int, int]]
        List of qubit pairs to swap.
    """
    for (i, j) in pairs:
        qc.swap(i, j)


def real_amplitudes_cz_extended(
    num_qubits: int,
    reps: int = 1,
    max_depth: int | None = None,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
    skip_final_rotation_layer: bool = False,
    insert_barriers: bool = False,
    add_swaps: bool = False,
    swap_schedule: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "circular",
    parameter_prefix: str = "theta",
    name: str | None = None,
) -> QuantumCircuit:
    """
    Construct a depth‑controlled hybrid ansatz based on RealAmplitudesCZ.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    reps : int, optional
        Number of CZ‑entanglement repetitions. Ignored if ``max_depth`` is set.
    max_depth : int | None, optional
        If provided, overrides ``reps`` and sets the total number of rotation layers.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]], optional
        Entanglement schedule.
    skip_final_rotation_layer : bool, optional
        If True, no rotation layer is appended after the last entanglement block.
    insert_barriers : bool, optional
        If True, insert barriers between logical layers for easier debugging.
    add_swaps : bool, optional
        If True, insert a SWAP network after each entanglement block.
    swap_schedule : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]], optional
        Schedule of SWAP pairs when ``add_swaps`` is True.
    parameter_prefix : str, optional
        Prefix for the rotation parameters.
    name : str | None, optional
        Circuit name. Defaults to ``'RealAmplitudesCZExtended'``.

    Returns
    -------
    QuantumCircuit
        The constructed ansatz circuit.
    """
    if num_qubits < 1:
        raise ValueError("num_qubits must be >= 1.")
    if reps < 0:
        raise ValueError("reps must be non‑negative.")
    if max_depth is not None and max_depth < 1:
        raise ValueError("max_depth must be >= 1 if provided.")

    depth = max_depth if max_depth is not None else reps
    n = int(num_qubits)

    qc = QuantumCircuit(n, name=name or "RealAmplitudesCZExtended")

    # Determine number of rotation layers: one before each entanglement block and
    # optionally one final rotation layer.
    num_rot_layers = depth if skip_final_rotation_layer else depth + 1
    params = ParameterVector(parameter_prefix, num_rot_layers * n)

    def _rot(layer: int) -> None:
        base = layer * n
        for q in range(n):
            qc.ry(params[base + q], q)

    pairs = _resolve_entanglement(n, entanglement)

    swap_pairs: List[Tuple[int, int]] = []
    if add_swaps:
        swap_pairs = _resolve_entanglement(n, swap_schedule)

    for r in range(depth):
        _rot(r)
        if insert_barriers:
            qc.barrier()
        for (i, j) in pairs:
            qc.cz(i, j)
        if add_swaps:
            _apply_swaps(qc, swap_pairs)
        if insert_barriers:
            qc.barrier()

    if not skip_final_rotation_layer:
        _rot(depth)

    qc.input_params = params  # type: ignore[attr-defined]
    qc.num_rot_layers = num_rot_layers  # type: ignore[attr-defined]
    return qc


class RealAmplitudesCZExtended(QuantumCircuit):
    """Convenience wrapper for RealAmplitudesCZExtended ansatz.

    Parameters are identical to :func:`real_amplitudes_cz_extended`. The class
    simply builds the circuit using the convenience function and then composes
    it into ``self``. This allows for composition with other circuits and
    parameter binding via the standard Qiskit API.
    """

    def __init__(
        self,
        num_qubits: int,
        reps: int = 1,
        max_depth: int | None = None,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
        skip_final_rotation_layer: bool = False,
        insert_barriers: bool = False,
        add_swaps: bool = False,
        swap_schedule: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "circular",
        parameter_prefix: str = "theta",
        name: str = "RealAmplitudesCZExtended",
    ) -> None:
        built = real_amplitudes_cz_extended(
            num_qubits,
            reps,
            max_depth,
            entanglement,
            skip_final_rotation_layer,
            insert_barriers,
            add_swaps,
            swap_schedule,
            parameter_prefix,
            name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)
        self.input_params = built.input_params  # type: ignore[attr-defined]
        self.num_rot_layers = built.num_rot_layers  # type: ignore[attr-defined]


__all__ = ["RealAmplitudesCZExtended", "real_amplitudes_cz_extended"]
