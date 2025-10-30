"""
RealAmplitudesExtended ansatz builder.

This module implements an extended version of the standard RealAmplitudes
ansatz (RY rotations + CX entanglers).  It adds:
- a configurable `depth` of full RY‑CX cycles,
- optional `mid_entanglement` that applies the entanglement schedule before
  the rotation of each cycle,
- a `hybrid_block` that can replace the default CX with a custom two‑qubit
  gate (e.g. a parameterized RZZ).  When a hybrid block is supplied,
  a dedicated parameter vector is created for the two‑qubit gates,
- optional barriers after each sub‑block.

The function `real_amplitudes_extended` returns a `QuantumCircuit` and the
class `RealAmplitudesExtended` behaves like Qiskit's native ansatz classes.
"""

from __future__ import annotations

from typing import Callable, List, Sequence, Tuple, Union, Optional

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.parameter import Parameter


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


def real_amplitudes_extended(
    num_qubits: int,
    depth: int = 1,
    mid_entanglement: bool = False,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
    skip_final_rotation_layer: bool = False,
    insert_barriers: bool = False,
    hybrid_block: Callable[[QuantumCircuit, int, int, Parameter], None] | None = None,
    parameter_prefix: str = "theta",
    name: str | None = None,
) -> QuantumCircuit:
    """Build an extended RealAmplitudes ansatz.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    depth : int, default 1
        Number of full RY‑CX cycles.  Each cycle contains one RY rotation layer
        followed by an entanglement schedule.
    mid_entanglement : bool, default False
        If True, apply the entanglement schedule *before* the RY layer of each
        cycle.  If False, the entanglement is applied *after* the RY layer.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]]
        Specification of the two‑qubit entanglement schedule.  See
        :func:`_resolve_entanglement` for details.
    skip_final_rotation_layer : bool, default False
        If True, do not add a rotation layer after the last cycle.
    insert_barriers : bool, default False
        Insert a barrier after each sub‑block (entanglement or rotation).
    hybrid_block : Callable[[QuantumCircuit, int, int, Parameter], None] | None, default None
        Replace the default CX gate with a custom two‑qubit gate.  The callable
        receives the circuit, the two qubit indices and a Parameter object
        that should be used for a parameterized gate (e.g. RZZ).  When
        ``None`` the standard CX is used.
    parameter_prefix : str, default "theta"
        Prefix for the RY rotation parameters.  If ``hybrid_block`` is provided,
        an additional prefix ``parameter_prefix + "_hz"`` is used.
    name : str | None, default None
        Name of the constructed circuit.

    Returns
    -------
    QuantumCircuit
        The constructed ansatz circuit.  The circuit has the attributes
        ``input_params`` (Ry parameters) and, if a hybrid block is used,
        ``input_params_hz`` (parameters for the two‑qubit gates).

    Raises
    ------
    ValueError
        If ``num_qubits`` is less than 1 or if ``depth`` is less than 1.
    """
    if num_qubits < 1:
        raise ValueError("num_qubits must be >= 1.")
    if depth < 1:
        raise ValueError("depth must be >= 1.")

    n = int(num_qubits)
    qc = QuantumCircuit(n, name=name or "RealAmplitudesExtended")

    # Rotation parameters
    num_rot_layers = depth if skip_final_rotation_layer else depth + 1
    rot_params = ParameterVector(parameter_prefix, num_rot_layers * n)

    def _rotation_layer(layer_idx: int) -> None:
        base = layer_idx * n
        for q in range(n):
            qc.ry(rot_params[base + q], q)

    pairs = _resolve_entanglement(n, entanglement)

    # Hybrid block parameters
    if hybrid_block is not None:
        num_hybrid_params = depth * len(pairs)
        hz_params = ParameterVector(parameter_prefix + "_hz", num_hybrid_params)
    else:
        hz_params = None

    for r in range(depth):
        if mid_entanglement:
            for pair_idx, (i, j) in enumerate(pairs):
                if hybrid_block is None:
                    qc.cx(i, j)
                else:
                    param = hz_params[r * len(pairs) + pair_idx]
                    hybrid_block(qc, i, j, param)
            if insert_barriers:
                qc.barrier()

        _rotation_layer(r)
        if insert_barriers:
            qc.barrier()

        if not mid_entanglement:
            for pair_idx, (i, j) in enumerate(pairs):
                if hybrid_block is None:
                    qc.cx(i, j)
                else:
                    param = hz_params[r * len(pairs) + pair_idx]
                    hybrid_block(qc, i, j, param)
            if insert_barriers:
                qc.barrier()

    if not skip_final_rotation_layer:
        _rotation_layer(depth)
        if insert_barriers:
            qc.barrier()

    qc.input_params = rot_params  # type: ignore[attr-defined]
    qc.num_rot_layers = num_rot_layers  # type: ignore[attr-defined]
    if hz_params is not None:
        qc.input_params_hz = hz_params  # type: ignore[attr-defined]
        qc.num_hybrid_layers = depth  # type: ignore[attr-defined]
    return qc


class RealAmplitudesExtended(QuantumCircuit):
    """Class‑style wrapper for the extended RealAmplitudes ansatz."""

    def __init__(
        self,
        num_qubits: int,
        depth: int = 1,
        mid_entanglement: bool = False,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
        skip_final_rotation_layer: bool = False,
        insert_barriers: bool = False,
        hybrid_block: Callable[[QuantumCircuit, int, int, Parameter], None] | None = None,
        parameter_prefix: str = "theta",
        name: str = "RealAmplitudesExtended",
    ) -> None:
        built = real_amplitudes_extended(
            num_qubits=num_qubits,
            depth=depth,
            mid_entanglement=mid_entanglement,
            entanglement=entanglement,
            skip_final_rotation_layer=skip_final_rotation_layer,
            insert_barriers=insert_barriers,
            hybrid_block=hybrid_block,
            parameter_prefix=parameter_prefix,
            name=name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)

        self.input_params = built.input_params  # type: ignore[attr-defined]
        self.num_rot_layers = built.num_rot_layers  # type: ignore[attr-defined]
        if hasattr(built, "input_params_hz"):
            self.input_params_hz = built.input_params_hz  # type: ignore[attr-defined]
            self.num_hybrid_layers = built.num_hybrid_layers  # type: ignore[attr-defined]


__all__ = ["RealAmplitudesExtended", "real_amplitudes_extended"]
