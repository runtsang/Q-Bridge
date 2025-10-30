"""RealAmplitudes variant with controlled symmetry and parameter‑sharing modifications.

This module implements a controlled‑modification of the original
`RealAmplitudesAlternating` ansatz.  Two optional knobs are provided:

* ``symmetry`` – enforce reflection symmetry in the rotation angles
  (parameter for qubit q equals that for qubit n‑q‑1).
* ``parameter_sharing`` – force all qubits in a single rotation layer to
  use the same parameter.

Both knobs are mutually exclusive.  When neither is enabled the ansatz
behaves identically to the seed.
"""

from __future__ import annotations

from math import ceil
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


def real_amplitudes_alternating_controlled_modification(
    num_qubits: int,
    reps: int = 1,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
    skip_final_rotation_layer: bool = False,
    insert_barriers: bool = False,
    parameter_prefix: str = "theta",
    symmetry: bool = False,
    parameter_sharing: bool = False,
    name: str | None = None,
) -> QuantumCircuit:
    """
    Build a RealAmplitudes alternating‑rotation ansatz with optional symmetry
    or parameter‑sharing constraints.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    reps : int, default 1
        Number of entangling‑rotation blocks.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]]
        Entanglement pattern.  ``"full"``, ``"linear"``, or ``"circular"``
        are accepted.  A custom sequence or callable may be supplied.
    skip_final_rotation_layer : bool, default False
        When ``True`` the final rotation layer after the last entanglement
        block is omitted.
    insert_barriers : bool, default False
        Insert a barrier between each logical block for easier debugging.
    parameter_prefix : str, default "theta"
        Prefix for the rotation parameters.
    symmetry : bool, default False
        When ``True`` enforce reflection symmetry: theta_q == theta_{n-q-1}
        for each rotation layer.
    parameter_sharing : bool, default False
        When ``True`` all qubits in a single rotation layer share the same
        parameter.  This is incompatible with ``symmetry``.
    name : str | None, default None
        Optional name for the circuit.  If omitted, a default name is
        constructed from the class name.

    Returns
    -------
    QuantumCircuit
        The constructed ansatz circuit.  The circuit has two attributes
        ``input_params`` (the ParameterVector) and ``num_rot_layers`` (the
        number of rotation layers).

    Raises
    ------
    ValueError
        If ``num_qubits`` < 1, or if both ``symmetry`` and
        ``parameter_sharing`` are ``True``.
    """
    if num_qubits < 1:
        raise ValueError("num_qubits must be >= 1.")
    if symmetry and parameter_sharing:
        raise ValueError("Parameters symmetry and sharing cannot both be enabled.")

    n = int(num_qubits)
    qc = QuantumCircuit(n, name=name or "RealAmplitudesCtrlMod")

    # Determine the number of rotation layers.
    num_rot_layers = reps if skip_final_rotation_layer else reps + 1

    # Parameters per layer
    if parameter_sharing:
        params_per_layer = 1
    elif symmetry:
        params_per_layer = (n + 1) // 2  # ceil(n/2)
    else:
        params_per_layer = n

    total_params = num_rot_layers * params_per_layer
    params = ParameterVector(parameter_prefix, total_params)

    def _rot(layer: int) -> None:
        base = layer * params_per_layer
        if layer % 2 == 0:  # RY layer
            for q in range(n):
                if parameter_sharing:
                    idx = base
                elif symmetry:
                    idx = base + min(q, n - 1 - q)
                else:
                    idx = base + q
                qc.ry(params[idx], q)
        else:  # RX layer
            for q in range(n):
                if parameter_sharing:
                    idx = base
                elif symmetry:
                    idx = base + min(q, n - 1 - q)
                else:
                    idx = base + q
                qc.rx(params[idx], q)

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


class RealAmplitudesAlternatingControlledModification(QuantumCircuit):
    """Convenience subclass for the controlled‑modification ansatz.

    The class simply constructs the underlying circuit using
    :func:`real_amplitudes_alternating_controlled_modification` and
    exposes the same public attributes as the base class.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    reps : int, default 1
        Number of entangling‑rotation blocks.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]]
        Entanglement pattern.
    skip_final_rotation_layer : bool, default False
        Skip the final rotation layer.
    insert_barriers : bool, default False
        Insert barriers between logical blocks.
    parameter_prefix : str, default "theta"
        Prefix for the rotation parameters.
    symmetry : bool, default False
        Enforce reflection symmetry.
    parameter_sharing : bool, default False
        Share parameters across qubits in a layer.
    name : str, default "RealAmplitudesAlternatingControlledModification"
        Name of the circuit.
    """

    def __init__(
        self,
        num_qubits: int,
        reps: int = 1,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
        skip_final_rotation_layer: bool = False,
        insert_barriers: bool = False,
        parameter_prefix: str = "theta",
        symmetry: bool = False,
        parameter_sharing: bool = False,
        name: str = "RealAmplitudesAlternatingControlledModification",
    ) -> None:
        built = real_amplitudes_alternating_controlled_modification(
            num_qubits,
            reps,
            entanglement,
            skip_final_rotation_layer,
            insert_barriers,
            parameter_prefix,
            symmetry,
            parameter_sharing,
            name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)
        self.input_params = built.input_params  # type: ignore[attr-defined]
        self.num_rot_layers = built.num_rot_layers  # type: ignore[attr-defined]


__all__ = [
    "RealAmplitudesAlternatingControlledModification",
    "real_amplitudes_alternating_controlled_modification",
]
