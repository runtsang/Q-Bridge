"""
RealAmplitudes variant using CZ entangling gates with depth‑controlled extension.

This module implements a *RealAmplitudesCZExtension* ansatz that expands on the
original RealAmplitudes CZ variant.  The new design introduces:
  * **Depth‑controlled entanglement** – a ``depth`` parameter that controls
    the number of entangling cycles per repetition.
  * **Adaptive entanglement schedules** – a callable that can generate a new
    schedule for each repetition.
  * **Optional controlled‑RZ (CRZ) gates** that act on each entangling pair.
  * **Barrier insertion**‑time, and parameter‑prefix customization.
The module offers a convenience function and a subclass that both
  * are Qiskit‑compatible (composition, parameter binding, etc.);
  * expose the same public name – *RealAmplitudesCZExtension*.

The design deliberately keeps the interface familiar while providing
additional knobs for richer expressivity.
"""

from __future__ import annotations

from typing import Callable, Iterable, List, Sequence, Tuple

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector


def _resolve_entanglement(
    num_qubits: int,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Iterable[Tuple[int, int]]],
) -> List[Tuple[int, int]]:
    """
    Resolve an entanglement specification into a list of qubit pairs.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Iterable[Tuple[int, int]]]
        Either a predefined schedule string or a custom list/callable.

    Returns
    -------
    List[Tuple[int, int]]
        Validated list of qubit pairs.

    Raises
    ------
    ValueError
        If an unknown entanglement string is supplied or a pair contains
        out‑of‑range qubits.
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
    else:
        pairs = list(entanglement)

    # Validation
    for (i, j) in pairs:
        if i == j:
            raise ValueError("Entanglement pairs must connect distinct qubits.")
        if not (0 <= i < num_qubits and 0 <= j < num_qubits):
            raise ValueError(f"Entanglement pair {(i, j)} out of range for n={num_qubits}.")
    return [(int(i), int(j)) for (i, j) in pairs]


def real_amplitudes_cz_extension(
    num_qubits: int,
    reps: int = 1,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Iterable[Tuple[int, int]]] = "full",
    depth: int = 1,
    skip_final_rotation_layer: bool = False,
    insert_barriers: bool = False,
    use_crz: bool = False,
    parameter_prefix: str = "theta",
    name: str | None = None,
) -> QuantumCircuit:
    """
    Construct a depth‑controlled RealAmplitudes CZ ansatz with optional CRZ gates.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the ansatz.
    reps : int, optional
        Number of rotation + entanglement repetitions.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Iterable[Tuple[int, int]]], optional
        Entanglement schedule. See :func:`_resolve_entanglement` for options.
    depth : int, optional
        Number of entangling cycles per repetition.
    skip_final_rotation_layer : bool, optional
        If ``True``, omit the final rotation layer.
    insert_barriers : bool, optional
        Insert barriers between layers for visual clarity.
    use_crz : bool, optional
        If ``True``, insert a controlled‑RZ gate after each CZ on the same pair.
    parameter_prefix : str, optional
        Prefix for rotation parameters.
    name : str | None, optional
        Name of the circuit.

    Returns
    -------
    QuantumCircuit
        The constructed ansatz circuit.

    Raises
    ------
    ValueError
        If ``num_qubits`` or ``depth`` is less than 1, or ``reps`` is less than 1.
    """
    if num_qubits < 1:
        raise ValueError("num_qubits must be >= 1.")
    if reps < 1:
        raise ValueError("reps must be >= 1.")
    if depth < 1:
        raise ValueError("depth must be >= 1.")

    n = int(num_qubits)
    qc = QuantumCircuit(n, name=name or "RealAmplitudesCZExtension")

    num_rot_layers = reps if skip_final_rotation_layer else reps + 1
    params = ParameterVector(parameter_prefix, num_rot_layers * n)

    # CRZ parameters, if requested
    crz_params: ParameterVector | None = None
    if use_crz:
        num_crz = reps * depth * len(_resolve_entanglement(n, entanglement))
        crz_params = ParameterVector(f"{parameter_prefix}_crz", num_crz)

    def _rot(layer: int) -> None:
        base = layer * n
        for q in range(n):
            qc.ry(params[base + q], q)

    pairs = _resolve_entanglement(n, entanglement)

    for r in range(reps):
        _rot(r)
        if insert_barriers:
            qc.barrier()
        for d in range(depth):
            for (i, j) in pairs:
                qc.cz(i, j)
                if use_crz and crz_params is not None:
                    # compute index into crz_params
                    pair_index = pairs.index((i, j))
                    crz_idx = r * depth * len(pairs) + d * len(pairs) + pair_index
                    qc.crz(crz_params[crz_idx], i, j)
            if insert_barriers:
                qc.barrier()

    if not skip_final_rotation_layer:
        _rot(reps)

    # Attach metadata
    qc.input_params = params  # type: ignore[attr-defined]
    qc.num_rot_layers = num_rot_layers  # type: ignore[attr-defined]
    if use_crz:
        qc.crz_params = crz_params  # type: ignore[attr-defined]

    return qc


class RealAmplitudesCZExtension(QuantumCircuit):
    """
    Class wrapper for the depth‑controlled CZ‑entangling RealAmplitudes ansatz.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the ansatz.
    reps : int, optional
        Number of rotation + entanglement repetitions.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Iterable[Tuple[int, int]]], optional
        Entanglement schedule. See :func:`_resolve_entanglement` for options.
    depth : int, optional
        Number of entangling cycles per repetition.
    skip_final_rotation_layer : bool, optional
        If ``True``, omit the final rotation layer.
    insert_barriers : bool, optional
        Insert barriers between layers for visual clarity.
    use_crz : bool, optional
        If ``True``, insert a controlled‑RZ gate after each CZ on the same pair.
    parameter_prefix : str, optional
        Prefix for rotation parameters.
    name : str, optional
        Name of the circuit.

    Notes
    -----
    The class inherits from :class:`qiskit.QuantumCircuit` and therefore
    supports all standard Qiskit operations such as parameter binding,
    circuit composition, and simulation.  It exposes the same metadata
    attributes as the functional interface: ``input_params``,
    ``num_rot_layers``, and (when ``use_crz`` is ``True``) ``crz_params``.
    """

    def __init__(
        self,
        num_qubits: int,
        reps: int = 1,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Iterable[Tuple[int, int]]] = "full",
        depth: int = 1,
        skip_final_rotation_layer: bool = False,
        insert_barriers: bool = False,
        use_crz: bool = False,
        parameter_prefix: str = "theta",
        name: str = "RealAmplitudesCZExtension",
    ) -> None:
        built = real_amplitudes_cz_extension(
            num_qubits,
            reps,
            entanglement,
            depth,
            skip_final_rotation_layer,
            insert_barriers,
            use_crz,
            parameter_prefix,
            name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)
        self.input_params = built.input_params  # type: ignore[attr-defined]
        self.num_rot_layers = built.num_rot_layers  # type: ignore[attr-defined]
        if use_crz:
            self.crz_params = built.crz_params  # type: ignore[attr-defined]


__all__ = ["RealAmplitudesCZExtension", "real_amplitudes_cz_extension"]
