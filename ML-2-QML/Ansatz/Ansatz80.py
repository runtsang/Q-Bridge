"""
RealAmplitudes CZ variant with controlled rotations.

This module implements a quantum circuit ansatz that is
identical in interface to :func:`real_amplitudes_cz` in the
original repository, but replaces each single‑qubit rotation
with a controlled rotation that depends on the previous qubit.
The change is a *controlled modification*: the parameter count
and overall structure remain unchanged, only the internal
gate pattern is altered to introduce a parity‑conserving
symmetry.

The ansatz is useful when one wishes to enforce a
controlled‑rotational structure, for example, in variational
algorithms that benefit from a more constrained entanglement
scheme.
"""

from __future__ import annotations

from typing import Callable, List, Sequence, Tuple

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector


def _resolve_entanglement(
    num_qubits: int,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]],
) -> List[Tuple[int, int]]:
    """
    Resolve an entanglement specification into a list of qubit pairs.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]]
        Entanglement specification.  Strings are interpreted as
        ``"full"``, ``"linear"``, or ``"circular"``.  Callables are
        expected to return a sequence of pairs given ``num_qubits``.
        Sequences are used verbatim.

    Returns
    -------
    List[Tuple[int, int]]
        List of unique, valid qubit pairs.

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
    name: str | None = None,
    controlled_rotations: bool = True,
) -> QuantumCircuit:
    """
    Construct a Real‑Amplitudes ansatz with CZ entanglement and optional
    controlled rotations.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the ansatz.
    reps : int, optional
        Number of repetition layers.  Default is ``1``.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]], optional
        Specification of which qubit pairs receive a CZ gate.  The
        default ``"full"`` applies CZ to every distinct pair.
    skip_final_rotation_layer : bool, optional
        If ``True`` the last single‑qubit rotation layer is omitted.
    insert_barriers : bool, optional
        If ``True`` a barrier is inserted after each rotation and
        entanglement block.
    parameter_prefix : str, optional
        Prefix for the parameter vector.  Default ``"theta"``.
    name : str | None, optional
        Circuit name.  If ``None`` the default name is used.
    controlled_rotations : bool, optional
        If ``True`` each RY rotation after the first qubit is replaced
        by a CRY gate controlled by the preceding qubit.  This
        introduces a parity‑conserving rotational structure.
        Default is ``True``; set to ``False`` to recover the original
        RY pattern.

    Returns
    -------
    QuantumCircuit
        The constructed ansatz circuit.

    Raises
    ------
    ValueError
        If ``num_qubits`` is less than 1.
    """
    if num_qubits < 1:
        raise ValueError("num_qubits must be >= 1.")

    n = int(num_qubits)
    qc = QuantumCircuit(n, name=name or "RealAmplitudesCZControlled")

    # Determine the number of rotation layers
    num_rot_layers = reps if skip_final_rotation_layer else reps + 1
    # Parameter vector: one parameter per qubit per rotation layer
    params = ParameterVector(parameter_prefix, num_rot_layers * n)

    # Rotation layer definition
    def _rot(layer: int) -> None:
        base = layer * n
        if controlled_rotations:
            # First qubit: ordinary RY
            qc.ry(params[base], 0)
            # Subsequent qubits: CRY controlled by previous qubit
            for q in range(1, n):
                qc.cry(params[base + q], q - 1, q)
        else:
            # Standard RY on all qubits
            for q in range(n):
                qc.ry(params[base + q], q)

    # Resolve entanglement schedule
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


class RealAmplitudesCZControlled(QuantumCircuit):
    """
    Convenience wrapper for the controlled‑rotation Real‑Amplitudes CZ ansatz.

    Parameters
    ----------
    num_qubits : int
        Number of qubits.
    reps : int, optional
        Number of repetition layers.  Default ``1``.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]], optional
        Entanglement specification.  Default ``"full"``.
    skip_final_rotation_layer : bool, optional
        Whether to omit the final rotation layer.  Default ``False``.
    insert_barriers : bool, optional
        Whether to insert barriers.  Default ``False``.
    parameter_prefix : str, optional
        Prefix for the parameter vector.  Default ``"theta"``.
    name : str, optional
        Circuit name.  Default ``"RealAmplitudesCZControlled"``.
    controlled_rotations : bool, optional
        Whether to use controlled rotations.  Default ``True``.
    """

    def __init__(
        self,
        num_qubits: int,
        reps: int = 1,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
        skip_final_rotation_layer: bool = False,
        insert_barriers: bool = False,
        parameter_prefix: str = "theta",
        name: str = "RealAmplitudesCZControlled",
        controlled_rotations: bool = True,
    ) -> None:
        built = real_amplitudes_cz_controlled(
            num_qubits,
            reps,
            entanglement,
            skip_final_rotation_layer,
            insert_barriers,
            parameter_prefix,
            name,
            controlled_rotations,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)
        self.input_params = built.input_params  # type: ignore[attr-defined]
        self.num_rot_layers = built.num_rot_layers  # type: ignore[attr-defined]


__all__ = ["RealAmplitudesCZControlled", "real_amplitudes_cz_controlled"]
