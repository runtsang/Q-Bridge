"""RealAmplitudesCZExtended – a depth‑controlled, hybrid‑layer Ansatz for Qiskit.

The implementation extends the classic Real‑Amplitudes ansatz that uses CZ
entanglers.  An optional **mix layer** is inserted after each CZ block.
The mix layer applies per‑qubit RZ rotations parameterised by a separate
parameter vector (`mix_params`).  The depth of the overall circuit can be
scaled by `depth_factor`, and the mix layer can be repeated `mix_depth`
times per repetition.  All extension parameters default to values that
leave the original behaviour unchanged.

Typical usage:

>>> qc = real_amplitudes_cz_extended(num_qubits=4, reps=3, mix_layer=True,
...                                 mix_depth=2, mix_factor=0.5)
>>> qc.draw()
"""
from __future__ import annotations

from typing import Callable, List, Sequence, Tuple, Union

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector


def _resolve_entanglement(
    num_qubits: int,
    entanglement: Union[
        str,
        Sequence[Tuple[int, int]],
        Callable[[int], Sequence[Tuple[int, int]]],
    ],
) -> List[Tuple[int, int]]:
    """Return a list of two‑qubit pairs according to a simple entanglement
    specification.

    The accepted string values are ``"full"``, ``"linear"``, and
    ``"circular"``.  A user‑defined callable or a concrete sequence of
    tuples can also be supplied.  The function performs basic sanity
    checks on the resulting pairs.
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
        return [(int(i), int(j)) for i, j in pairs]

    pairs = [(int(i), int(j)) for i, j in entanglement]  # type: ignore[arg-type]
    for i, j in pairs:
        if i == j:
            raise ValueError("Entanglement pairs must connect distinct qubits.")
        if not (0 <= i < num_qubits and 0 <= j < num_qubits):
            raise ValueError(f"Entanglement pair {(i, j)} out of range for n={num_qubits}.")
    return pairs


def real_amplitudes_cz_extended(
    num_qubits: int,
    reps: int = 1,
    entanglement: Union[
        str,
        Sequence[Tuple[int, int]],
        Callable[[int], Sequence[Tuple[int, int]]],
    ] = "full",
    skip_final_rotation_layer: bool = False,
    insert_barriers: bool = False,
    parameter_prefix: str = "theta",
    mix_layer: bool = False,
    mix_entanglement: Union[
        str,
        Sequence[Tuple[int, int]],
        Callable[[int], Sequence[Tuple[int, int]]],
    ] = "full",
    mix_parameter_prefix: str = "phi",
    mix_depth: int = 1,
    mix_factor: float = 1.0,
    depth_factor: int = 1,
    name: str | None = None,
) -> QuantumCircuit:
    """
    Build a Real‑Amplitudes CZ ansatz with optional depth and mix‑layer
    extensions.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.  Must be >= 1.
    reps : int
        Base repetition count of the rotation‑entanglement cycle.  Must be
        >= 0.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]]
        Specification of the CZ entanglement schedule.  Standard strings
        are ``"full"``, ``"linear"``, and ``"circular"``.  A callable or
        explicit list of tuples is also accepted.
    skip_final_rotation_layer : bool
        If ``True`` the final rotation layer (after the last CZ block)
        is omitted.
    insert_barriers : bool
        If ``True`` a barrier is inserted after each rotation and
        entanglement block for visual clarity.
    parameter_prefix : str
        Prefix for the rotation parameters.
    mix_layer : bool
        If ``True`` an additional mix layer consisting of per‑qubit RZ
        rotations is inserted after each CZ block.
    mix_entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]]
        Entanglement schedule used to compute the number of mix
        parameters.  It defaults to ``"full"`` and follows the same
        semantics as ``entanglement``.
    mix_parameter_prefix : str
        Prefix for the mix‑layer parameters.
    mix_depth : int
        Number of times the mix layer is repeated per CZ block.  Must be
        >= 0 and ignored if ``mix_layer`` is ``False``.
    mix_factor : float
        Global scaling factor applied to all mix‑layer RZ angles.
    depth_factor : int
        Factor by which the base repetition count is multiplied.
        ``depth_factor`` >= 1.
    name : str | None
        Optional name for the circuit.  Defaults to
        ``"RealAmplitudesCZExtended"``.

    Returns
    -------
    QuantumCircuit
        The constructed ansatz circuit.  The circuit exposes the
        attributes ``input_params`` (rotation parameters),
        ``mix_params`` (mix‑layer parameters, if any),
        ``num_rot_layers`` and ``num_mix_layers`` for introspection.
    """
    if num_qubits < 1:
        raise ValueError("num_qubits must be >= 1.")
    if reps < 0:
        raise ValueError("reps must be >= 0.")
    if depth_factor < 1:
        raise ValueError("depth_factor must be >= 1.")
    if mix_layer and mix_depth < 0:
        raise ValueError("mix_depth must be >= 0.")
    if mix_layer and mix_depth == 0:
        # If mix_depth is zero, treat mix_layer as disabled
        mix_layer = False

    n = int(num_qubits)
    total_reps = reps * depth_factor
    num_rot_layers = total_reps if skip_final_rotation_layer else total_reps + 1

    qc = QuantumCircuit(n, name=name or "RealAmplitudesCZExtended")

    # Rotation parameters
    rot_params = ParameterVector(parameter_prefix, num_rot_layers * n)

    # Mix parameters, if enabled
    mix_params: ParameterVector | None = None
    if mix_layer:
        # The number of mix parameters equals mix_depth * total_reps * n
        mix_params = ParameterVector(
            mix_parameter_prefix, mix_depth * total_reps * n
        )

    def _rot(layer: int) -> None:
        base = layer * n
        for q in range(n):
            qc.ry(rot_params[base + q], q)

    def _mix(layer: int, sublayer: int) -> None:
        """Apply a mix layer consisting of RZ rotations."""
        base = (layer * mix_depth + sublayer) * n
        for q in range(n):
            qc.rz(mix_factor * mix_params[base + q], q)

    pairs = _resolve_entanglement(n, entanglement)

    for r in range(total_reps):
        _rot(r)
        if insert_barriers:
            qc.barrier()
        for (i, j) in pairs:
            qc.cz(i, j)
        if insert_barriers:
            qc.barrier()
        if mix_layer:
            for m in range(mix_depth):
                _mix(r, m)

    if not skip_final_rotation_layer:
        _rot(total_reps)

    qc.input_params = rot_params  # type: ignore[attr-defined]
    qc.num_rot_layers = num_rot_layers  # type: ignore[attr-defined]
    if mix_layer:
        qc.mix_params = mix_params  # type: ignore[attr-defined]
        qc.num_mix_layers = mix_depth * total_reps  # type: ignore[attr-defined]
    return qc


class RealAmplitudesCZExtended(QuantumCircuit):
    """Convenience subclass wrapping :func:`real_amplitudes_cz_extended`."""

    def __init__(
        self,
        num_qubits: int,
        reps: int = 1,
        entanglement: Union[
            str,
            Sequence[Tuple[int, int]],
            Callable[[int], Sequence[Tuple[int, int]]],
        ] = "full",
        skip_final_rotation_layer: bool = False,
        insert_barriers: bool = False,
        parameter_prefix: str = "theta",
        mix_layer: bool = False,
        mix_entanglement: Union[
            str,
            Sequence[Tuple[int, int]],
            Callable[[int], Sequence[Tuple[int, int]]],
        ] = "full",
        mix_parameter_prefix: str = "phi",
        mix_depth: int = 1,
        mix_factor: float = 1.0,
        depth_factor: int = 1,
        name: str = "RealAmplitudesCZExtended",
    ) -> None:
        built = real_amplitudes_cz_extended(
            num_qubits,
            reps,
            entanglement,
            skip_final_rotation_layer,
            insert_barriers,
            parameter_prefix,
            mix_layer,
            mix_entanglement,
            mix_parameter_prefix,
            mix_depth,
            mix_factor,
            depth_factor,
            name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)
        self.input_params = built.input_params  # type: ignore[attr-defined]
        self.num_rot_layers = built.num_rot_layers  # type: ignore[attr-defined]
        if mix_layer:
            self.mix_params = built.mix_params  # type: ignore[attr-defined]
            self.num_mix_layers = built.num_mix_layers  # type: ignore[attr-defined]


__all__ = ["RealAmplitudesCZExtended", "real_amplitudes_cz_extended"]
