"""RealAmplitudes variant using CZ entanglers with optional symmetry‑preserving parameter sharing.

This module implements a controlled‑modification of the original RealAmplitudesCZ ansatz.
A new boolean flag ``parameter_sharing`` controls whether the rotation parameters are shared
across all layers. When enabled, the circuit is invariant under qubit permutations
and uses only ``num_qubits`` parameters, which can be advantageous for symmetric data.

The public API consists of:
- ``real_amplitudes_cz_controlled``: convenience constructor.
- ``RealAmplitudesCZControlled``: subclass of ``QuantumCircuit`` that builds the ansatz.
"""

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


def real_amplitudes_cz_controlled(
    num_qubits: int,
    reps: int = 1,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
    skip_final_rotation_layer: bool = False,
    insert_barriers: bool = False,
    parameter_prefix: str = "theta",
    parameter_sharing: bool = True,
    name: str | None = None,
) -> QuantumCircuit:
    """Construct a RealAmplitudes ansatz with CZ entanglers and optional shared parameters.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the ansatz.
    reps : int
        Number of repeated rotation + entanglement layers.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]]
        Entanglement pattern. Supports the strings ``"full"``, ``"linear"``, ``"circular"``
        or a custom sequence/callable returning pairs of qubit indices.
    skip_final_rotation_layer : bool
        If ``True`` the final rotation layer after the last entangling block is omitted.
    insert_barriers : bool
        If ``True`` a barrier is inserted after each rotation and entangling block.
    parameter_prefix : str
        Prefix used for the parameter names in the circuit.
    parameter_sharing : bool
        When ``True`` all rotation layers share the same set of parameters,
        reducing the parameter count to ``num_qubits``. When ``False`` the
        original behaviour (unique parameters per layer) is preserved.
    name : str | None
        Optional name for the quantum circuit.

    Returns
    -------
    QuantumCircuit
        The constructed ansatz circuit.  The attributes ``input_params`` and
        ``num_rot_layers`` are attached for convenience.
    """
    if num_qubits < 1:
        raise ValueError("num_qubits must be >= 1.")
    if reps < 0:
        raise ValueError("reps must be >= 0.")

    n = int(num_qubits)
    qc = QuantumCircuit(n, name=name or "RealAmplitudesCZControlled")

    num_rot_layers = reps if skip_final_rotation_layer else reps + 1

    if parameter_sharing:
        params = ParameterVector(parameter_prefix, n)
    else:
        params = ParameterVector(parameter_prefix, num_rot_layers * n)

    def _rot(layer: int, base: int) -> None:
        """Apply a layer of RY rotations."""
        for q in range(n):
            qc.ry(params[base + q], q)

    pairs = _resolve_entanglement(n, entanglement)

    for r in range(reps):
        if parameter_sharing:
            _rot(r, 0)
        else:
            _rot(r, r * n)
        if insert_barriers:
            qc.barrier()
        for (i, j) in pairs:
            qc.cz(i, j)
        if insert_barriers:
            qc.barrier()

    if not skip_final_rotation_layer:
        if parameter_sharing:
            _rot(reps, 0)
        else:
            _rot(reps, reps * n)

    qc.input_params = params  # type: ignore[attr-defined]
    qc.num_rot_layers = num_rot_layers  # type: ignore[attr-defined]
    return qc


class RealAmplitudesCZControlled(QuantumCircuit):
    """Convenience wrapper subclass for the CZ‑entangling RealAmplitudes ansatz.

    The constructor forwards all arguments to :func:`real_amplitudes_cz_controlled`
    and then composes the resulting circuit into the instance.  The attributes
    ``input_params`` and ``num_rot_layers`` are copied for compatibility with
    other components that expect them.
    """

    def __init__(
        self,
        num_qubits: int,
        reps: int = 1,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
        skip_final_rotation_layer: bool = False,
        insert_barriers: bool = False,
        parameter_prefix: str = "theta",
        parameter_sharing: bool = True,
        name: str = "RealAmplitudesCZControlled",
    ) -> None:
        built = real_amplitudes_cz_controlled(
            num_qubits,
            reps,
            entanglement,
            skip_final_rotation_layer,
            insert_barriers,
            parameter_prefix,
            parameter_sharing,
            name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)
        self.input_params = built.input_params  # type: ignore[attr-defined]
        self.num_rot_layers = built.num_rot_layers  # type: ignore[attr-defined]


__all__ = ["RealAmplitudesCZControlled", "real_amplitudes_cz_controlled"]
