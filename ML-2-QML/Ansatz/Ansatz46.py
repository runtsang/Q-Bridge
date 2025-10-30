"""
RealAmplitudesAlternatingSymmetry: a symmetry‑constrained variant of the alternating‑rotation RealAmplitudes ansatz.
This implementation enforces that each entangled qubit pair shares a common rotation angle within each layer.
The ansatz is compatible with Qiskit’s QuantumCircuit API and supports parameter binding, composition, and classical simulation.
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
        Entanglement specification. For this symmetry‑constrained ansatz only
        the string ``"linear"`` and custom pair lists are supported.  The
        specification must form a perfect matching: each qubit appears exactly
        once across all pairs.

    Returns
    -------
    List[Tuple[int, int]]
        A list of (control, target) qubit indices.

    Raises
    ------
    ValueError
        If the specification is unsupported or does not form a perfect matching.
    """
    if isinstance(entanglement, str):
        if entanglement == "linear":
            if num_qubits % 2!= 0:
                raise ValueError(
                    "Linear entanglement requires an even number of qubits for a perfect matching."
                )
            return [(i, i + 1) for i in range(0, num_qubits, 2)]
        else:
            raise ValueError(
                f"Unsupported entanglement string {entanglement!r}. Only 'linear' is supported for symmetry‑constrained ansatz."
            )

    if callable(entanglement):
        pairs = list(entanglement(num_qubits))
    else:
        pairs = list(entanglement)

    # Validate pairs
    used_qubits = set()
    for (i, j) in pairs:
        if i == j:
            raise ValueError("Entanglement pairs must connect distinct qubits.")
        if not (0 <= i < num_qubits and 0 <= j < num_qubits):
            raise ValueError(f"Entanglement pair {(i, j)} out of range for n={num_qubits}.")
        used_qubits.update({i, j})

    if len(used_qubits)!= num_qubits:
        raise ValueError(
            "Entanglement pairs must form a perfect matching; each qubit must appear exactly once."
        )

    return [(int(i), int(j)) for (i, j) in pairs]


def real_amplitudes_alternating_symmetry(
    num_qubits: int,
    reps: int = 1,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "linear",
    skip_final_rotation_layer: bool = False,
    insert_barriers: bool = False,
    parameter_prefix: str = "theta",
    name: str | None = None,
) -> QuantumCircuit:
    """
    Construct a symmetry‑constrained alternating‑rotation RealAmplitudes ansatz.

    The ansatz alternates between RY and RX rotations on each layer, but
    enforces that every qubit pair defined by the entanglement schedule
    shares a single parameter per layer.  This reduces the number of
    trainable parameters and imposes a pairwise symmetry that can be
    beneficial for physics‑motivated circuits.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    reps : int, default 1
        Number of entanglement‑rotation layers.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]], default "linear"
        Entanglement schedule.  Only "linear" (perfect matching) or a custom
        list of pairs that form a perfect matching are supported.
    skip_final_rotation_layer : bool, default False
        If True, omit the final rotation layer after the last entanglement.
    insert_barriers : bool, default False
        If True, insert barriers between logical blocks for clarity.
    parameter_prefix : str, default "theta"
        Prefix for the parameter vector names.
    name : str | None, default None
        Optional name for the resulting QuantumCircuit.

    Returns
    -------
    QuantumCircuit
        The constructed ansatz circuit.

    Raises
    ------
    ValueError
        If `num_qubits` < 1, `reps` < 0, or the entanglement specification
        does not form a perfect matching.
    """
    if num_qubits < 1:
        raise ValueError("num_qubits must be >= 1.")
    if reps < 0:
        raise ValueError("reps must be non‑negative.")

    n = int(num_qubits)
    qc = QuantumCircuit(n, name=name or "RealAmplitudesAlternatingSymmetry")

    # Resolve entanglement and validate perfect matching
    pairs = _resolve_entanglement(n, entanglement)
    num_pairs = len(pairs)

    num_rot_layers = reps if skip_final_rotation_layer else reps + 1
    params = ParameterVector(parameter_prefix, num_rot_layers * num_pairs)

    def _rot(layer: int) -> None:
        base = layer * num_pairs
        for p, (i, j) in enumerate(pairs):
            param = params[base + p]
            if layer % 2 == 0:
                qc.ry(param, i)
                qc.ry(param, j)
            else:
                qc.rx(param, i)
                qc.rx(param, j)

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
    """
    Class wrapper for the symmetry‑constrained alternating‑rotation RealAmplitudes ansatz.

    The constructor simply builds the circuit using :func:`real_amplitudes_alternating_symmetry`
    and composes it into the current instance.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    reps : int, default 1
        Number of entanglement‑rotation layers.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]], default "linear"
        Entanglement schedule.  Only "linear" (perfect matching) or a custom
        list of pairs that form a perfect matching are supported.
    skip_final_rotation_layer : bool, default False
        If True, omit the final rotation layer after the last entanglement.
    insert_barriers : bool, default False
        If True, insert barriers between logical blocks for clarity.
    parameter_prefix : str, default "theta"
        Prefix for the parameter vector names.
    name : str, default "RealAmplitudesAlternatingSymmetry"
        Name of the circuit.
    """

    def __init__(
        self,
        num_qubits: int,
        reps: int = 1,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "linear",
        skip_final_rotation_layer: bool = False,
        insert_barriers: bool = False,
        parameter_prefix: str = "theta",
        name: str = "RealAmplitudesAlternatingSymmetry",
    ) -> None:
        built = real_amplitudes_alternating_symmetry(
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


__all__ = ["RealAmplitudesAlternatingSymmetry", "real_amplitudes_alternating_symmetry"]
