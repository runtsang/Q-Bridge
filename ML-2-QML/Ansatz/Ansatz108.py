"""RealAmplitudes variant with alternating RY/RX rotation layers and optional parameter sharing.

This variant introduces a controlled modification to the original RealAmplitudesAlternating ansatz:
- Parameters can be shared across all qubits within a rotation layer, reducing the number of degrees of freedom and enforcing a global rotational symmetry.
- Users may toggle the sharing via ``share_params_per_layer``. When disabled, the ansatz reverts to the original per‑qubit parameterization.
- The entanglement schedule, barrier insertion, final rotation toggle, and repetition count remain unchanged.

The module exposes:
- ``real_amplitudes_alternating_controlled_modification(...)``: convenience constructor returning a ``QuantumCircuit``.
- ``RealAmplitudesAlternatingControlledModification``: subclass of ``QuantumCircuit`` with the same name.
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
    entanglement: Union[
        str,
        Sequence[Tuple[int, int]],
        Callable[[int], Sequence[Tuple[int, int]]],
    ] = "full",
    skip_final_rotation_layer: bool = False,
    insert_barriers: bool = False,
    parameter_prefix: str = "theta",
    name: str | None = None,
    *,
    share_params_per_layer: bool = True,
) -> QuantumCircuit:
    """
    Build a RealAmplitudes ansatz with alternating RY/RX layers and optional
    parameter sharing across qubits within each rotation layer.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    reps : int, default=1
        Number of entanglement + rotation repetitions.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]], default="full"
        Specification of which qubit pairs to entangle.
    skip_final_rotation_layer : bool, default=False
        If True, omit the final rotation layer after the last entangling block.
    insert_barriers : bool, default=False
        If True, insert barriers between layers for easier debugging.
    parameter_prefix : str, default="theta"
        Prefix used for generated Parameter objects.
    name : str | None, default=None
        Optional name for the circuit.
    share_params_per_layer : bool, default=True
        If True, all qubits in a given rotation layer share the same parameter.
        If False, each qubit has its own parameter (original behaviour).

    Returns
    -------
    QuantumCircuit
        The constructed parameterized ansatz circuit.
    """
    if num_qubits < 1:
        raise ValueError("num_qubits must be >= 1.")
    if reps < 0:
        raise ValueError("reps must be non‑negative.")
    if not isinstance(share_params_per_layer, bool):
        raise TypeError("share_params_per_layer must be a bool.")

    n = int(num_qubits)
    qc = QuantumCircuit(n, name=name or "RealAmplitudesAlternatingControlled")

    num_rot_layers = reps if skip_final_rotation_layer else reps + 1
    if share_params_per_layer:
        params = ParameterVector(parameter_prefix, num_rot_layers)
    else:
        params = ParameterVector(parameter_prefix, num_rot_layers * n)

    def _rot(layer: int) -> None:
        base = layer * n
        if layer % 2 == 0:
            # Even layers use RY
            if share_params_per_layer:
                for q in range(n):
                    qc.ry(params[layer], q)
            else:
                for q in range(n):
                    qc.ry(params[base + q], q)
        else:
            # Odd layers use RX
            if share_params_per_layer:
                for q in range(n):
                    qc.rx(params[layer], q)
            else:
                for q in range(n):
                    qc.rx(params[base + q], q)

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
    qc.share_params_per_layer = share_params_per_layer  # type: ignore[attr-defined]
    return qc


class RealAmplitudesAlternatingControlledModification(QuantumCircuit):
    """Class wrapper for the alternating‑rotation RealAmplitudes ansatz with optional parameter sharing."""

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
        name: str = "RealAmplitudesAlternatingControlled",
        *,
        share_params_per_layer: bool = True,
    ) -> None:
        built = real_amplitudes_alternating_controlled_modification(
            num_qubits,
            reps,
            entanglement,
            skip_final_rotation_layer,
            insert_barriers,
            parameter_prefix,
            name,
            share_params_per_layer=share_params_per_layer,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)
        self.input_params = built.input_params  # type: ignore[attr-defined]
        self.num_rot_layers = built.num_rot_layers  # type: ignore[attr-defined]
        self.share_params_per_layer = built.share_params_per_layer  # type: ignore[attr-defined]


__all__ = [
    "RealAmplitudesAlternatingControlledModification",
    "real_amplitudes_alternating_controlled_modification",
]
