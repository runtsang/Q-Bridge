"""
RealAmplitudesExtended ansatz (extension of the canonical RealAmplitudes).

This module implements a depth‑scaled, optionally parameter‑shared ansatz that
alternates RY rotation layers with a second RZ layer and follows each with a
CX entanglement schedule.  The design preserves the intuitive RY‑CX pattern
while providing richer expressivity.

Features
--------
- **Depth factor**: multiplies the nominal repetition count, allowing a
  fine‑grained control over expressivity without changing the user‑visible
  ``reps`` parameter.
- **Parameter sharing**: when enabled all rotation layers reuse the same
  parameters, reducing the total parameter count.
- **Second rotation layer**: a configurable RZ layer after each RY layer
  enhances entanglement without adding two‑qubit gates.
- **Custom entanglement schedules**: accepts the same ``entanglement`` spec
  as the original ansatz (``"full"``, ``"linear"``, ``"circular"``, or a
  custom sequence/callable).
- **Barrier support**: optional barriers for debugging or circuit
  segmentation.

Usage
-----
```python
from real_amplitudes_extended import real_amplitudes_extended, RealAmplitudesExtended

qc = real_amplitudes_extended(num_qubits=4, reps=2, depth_factor=1.5, share_parameters=True)
# or
qc = RealAmplitudesExtended(num_qubits=4, reps=2, depth_factor=1.5, share_parameters=True)
```
"""
from __future__ import annotations

from typing import Callable, List, Sequence, Tuple

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector


def _resolve_entanglement(
    num_qubits: int,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]],
) -> List[Tuple[int, int]]:
    """Translate an `entanglement` specification into a list of qubit pairs.

    Parameters
    ----------
    num_qubits
        Number of qubits in the circuit.
    entanglement
        * ``"full"``: all-to-all pairs.
        * ``"linear"``: adjacent pairs (i, i+1).
        * ``"circular"``: linear plus a pair connecting the last and first qubits.
        * custom sequence or callable returning a sequence of pairs.

    Returns
    -------
    List[Tuple[int, int]]
        Valid two‑qubit pairs for entanglement gates.

    Raises
    ------
    ValueError
        If an invalid specification or out‑of‑range pair is provided.
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

    # Sequence of pairs
    pairs = [(int(i), int(j)) for (i, j) in entanglement]  # type: ignore[arg-type]
    for (i, j) in pairs:
        if i == j:
            raise ValueError("Entanglement pairs must connect distinct qubits.")
        if not (0 <= i < num_qubits and 0 <= j < num_qubits):
            raise ValueError(f"Entanglement pair {(i, j)} out of range for n={num_qubits}.")
    return pairs


def real_amplitudes_extended(
    num_qubits: int,
    reps: int = 1,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
    skip_final_rotation_layer: bool = False,
    insert_barriers: bool = False,
    parameter_prefix: str = "theta",
    depth_factor: float = 1.0,
    share_parameters: bool = False,
    second_rotation: bool = True,
    name: str | None = None,
) -> QuantumCircuit:
    """Build an extended RealAmplitudes-style `QuantumCircuit`.

    Parameters
    ----------
    num_qubits
        Number of qubits in the ansatz.
    reps
        Nominal number of rotation–entanglement repetitions.
    entanglement
        Entanglement specification (see :func:`_resolve_entanglement`).
    skip_final_rotation_layer
        If ``True`` the final rotation layer is omitted.
    insert_barriers
        Insert barriers after each rotation and entanglement block.
    parameter_prefix
        Prefix for generated parameter names.
    depth_factor
        Multiplier applied to ``reps`` to compute the effective depth.
        Must be positive.
    share_parameters
        If ``True`` all rotation layers reuse the same parameter vector.
    second_rotation
        When ``True`` an additional RZ layer follows each RY layer.
    name
        Optional circuit name.

    Returns
    -------
    QuantumCircuit
        The constructed ansatz circuit.
    """
    if num_qubits < 1:
        raise ValueError("num_qubits must be >= 1.")
    if depth_factor <= 0:
        raise ValueError("depth_factor must be positive.")

    effective_reps = max(1, int(round(reps * depth_factor)))
    num_rot_layers = effective_reps if skip_final_rotation_layer else effective_reps + 1

    qc = QuantumCircuit(num_qubits, name=name or "RealAmplitudesExtended")

    # Parameter vectors
    if share_parameters:
        rot_params = ParameterVector(parameter_prefix + "_ry", num_qubits)
        rz_params = ParameterVector(parameter_prefix + "_rz", num_qubits) if second_rotation else None
    else:
        rot_params = ParameterVector(parameter_prefix + "_ry", num_rot_layers * num_qubits)
        rz_params = ParameterVector(parameter_prefix + "_rz", num_rot_layers * num_qubits) if second_rotation else None

    pairs = _resolve_entanglement(num_qubits, entanglement)

    def _apply_rotation_layer(layer_idx: int) -> None:
        for q in range(num_qubits):
            idx = q if share_parameters else layer_idx * num_qubits + q
            qc.ry(rot_params[idx], q)

    def _apply_second_rotation_layer(layer_idx: int) -> None:
        if not second_rotation:
            return
        for q in range(num_qubits):
            idx = q if share_parameters else layer_idx * num_qubits + q
            qc.rz(rz_params[idx], q)

    # Main repetition loop
    for r in range(effective_reps):
        _apply_rotation_layer(r)
        if insert_barriers:
            qc.barrier()
        _apply_second_rotation_layer(r)
        if insert_barriers:
            qc.barrier()
        for (i, j) in pairs:
            qc.cx(i, j)
        if insert_barriers:
            qc.barrier()

    # Final rotation layer (if not skipped)
    if not skip_final_rotation_layer:
        final_layer = effective_reps
        _apply_rotation_layer(final_layer)
        if insert_barriers:
            qc.barrier()
        _apply_second_rotation_layer(final_layer)
        if insert_barriers:
            qc.barrier()

    qc.input_params = (rot_params, rz_params)  # type: ignore[attr-defined]
    qc.num_rot_layers = num_rot_layers  # type: ignore[attr-defined]
    return qc


class RealAmplitudesExtended(QuantumCircuit):
    """Convenience subclass that constructs the extended ansatz in ``__init__``."""

    def __init__(
        self,
        num_qubits: int,
        reps: int = 1,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
        skip_final_rotation_layer: bool = False,
        insert_barriers: bool = False,
        parameter_prefix: str = "theta",
        depth_factor: float = 1.0,
        share_parameters: bool = False,
        second_rotation: bool = True,
        name: str = "RealAmplitudesExtended",
    ) -> None:
        built = real_amplitudes_extended(
            num_qubits=num_qubits,
            reps=reps,
            entanglement=entanglement,
            skip_final_rotation_layer=skip_final_rotation_layer,
            insert_barriers=insert_barriers,
            parameter_prefix=parameter_prefix,
            depth_factor=depth_factor,
            share_parameters=share_parameters,
            second_rotation=second_rotation,
            name=name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)
        self.input_params = built.input_params  # type: ignore[attr-defined]
        self.num_rot_layers = built.num_rot_layers  # type: ignore[attr-defined]


__all__ = ["RealAmplitudesExtended", "real_amplitudes_extended"]
