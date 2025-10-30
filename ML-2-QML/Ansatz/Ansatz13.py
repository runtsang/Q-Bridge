"""Extended RealAmplitudes ansatz builder.

This module implements an augmented version of the canonical RealAmplitudes
ansatz.  The new construction adds an optional RZ rotation layer after each
entangling CX block, thereby increasing the circuit depth and the number of
free parameters.  The interface mirrors the original builder so it can be
used interchangeably, but it exposes additional knobs for expressivity.
"""

from __future__ import annotations

from typing import Callable, List, Sequence, Tuple, Union

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

EntanglementSpec = Union[
    str,
    Sequence[Tuple[int, int]],
    Callable[[int], Sequence[Tuple[int, int]]],
]


def _resolve_entanglement(
    num_qubits: int,
    entanglement: EntanglementSpec,
) -> List[Tuple[int, int]]:
    """Return a list of qubit pairs for entanglement.

    The function accepts the same three kinds of specifications as the
    original RealAmplitudes builder:
    * a string identifying a predefined pattern ("full", "linear",
      "circular").
    * a user‑supplied sequence of pairs.
    * a callable that receives the number of qubits and returns a
      sequence of pairs.

    Validation checks that each pair contains distinct qubits and that
    all indices are within range.
    """
    if isinstance(entanglement, str):
        if entanglement == "full":
            return [
                (i, j) for i in range(num_qubits) for j in range(i + 1, num_qubits)
            ]
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

    # Assume sequence of pairs
    pairs = [(int(i), int(j)) for (i, j) in entanglement]
    for (i, j) in pairs:
        if i == j:
            raise ValueError("Entanglement pairs must connect distinct qubits.")
        if not (0 <= i < num_qubits and 0 <= j < num_qubits):
            raise ValueError(f"Entanglement pair {(i, j)} out of range for n={num_qubits}.")
    return pairs


def real_amplitudes_extended(
    num_qubits: int,
    reps: int = 1,
    entanglement: EntanglementSpec = "full",
    skip_final_rotation_layer: bool = False,
    insert_barriers: bool = False,
    parameter_prefix: str = "theta",
    rz_parameter_prefix: str = "phi",
    add_rz_after_entanglement: bool = False,
    name: str | None = None,
) -> QuantumCircuit:
    """Build an extended RealAmplitudes ansatz.

    The circuit consists of `reps` repetitions of
    * a layer of RY rotations,
    * an entanglement layer of CX gates (or a user‑supplied pattern),
    * optionally an RZ rotation layer if `add_rz_after_entanglement` is True.

    An additional final RY layer is added unless
    `skip_final_rotation_layer` is True.  The number of parameters is
    `reps * num_qubits` for the RY layer plus, if enabled,
    `reps * num_qubits` for the optional RZ layer.

    Parameters
    ----------
    num_qubits
        Number of qubits in the ansatz.
    reps
        Number of repeat blocks.  Must be >= 1.
    entanglement
        Specification of entanglement pairs.  See `_resolve_entanglement`
        for accepted values.
    skip_final_rotation_layer
        If True, the final RY rotation layer is omitted.
    insert_barriers
        If True, a barrier will be inserted before and after each
        entanglement block to aid debugging and visual clarity.
    parameter_prefix
        Prefix used for the RY rotation parameters.
    rz_parameter_prefix
        Prefix used for the optional RZ rotation parameters.
    add_rz_after_entanglement
        If True, an RZ rotation layer will be applied after each
        entanglement block.
    name
        Optional name for the constructed circuit.

    Returns
    -------
    QuantumCircuit
        The constructed ansatz.  The circuit exposes two attributes:
        ``input_params`` (ParameterVector for RY parameters) and,
        if requested, ``rz_input_params`` (ParameterVector for RZ
        parameters).  ``num_rot_layers`` records the number of RY
        rotation layers (including the optional final layer).
    """
    if num_qubits < 1:
        raise ValueError("num_qubits must be >= 1.")
    if reps < 1:
        raise ValueError("reps must be >= 1.")

    n = int(num_qubits)
    qc = QuantumCircuit(n, name=name or "RealAmplitudesExtended")

    # Total number of rotation layers (RY)
    num_rot_layers = reps if skip_final_rotation_layer else reps + 1
    params = ParameterVector(parameter_prefix, num_rot_layers * n)

    # Optional RZ parameters
    if add_rz_after_entanglement:
        num_rz_layers = reps
        rz_params = ParameterVector(rz_parameter_prefix, num_rz_layers * n)
    else:
        rz_params = None

    def _ry_layer(layer_idx: int) -> None:
        base = layer_idx * n
        for q in range(n):
            qc.ry(params[base + q], q)

    if add_rz_after_entanglement:

        def _rz_layer(layer_idx: int) -> None:
            base = layer_idx * n
            for q in range(n):
                qc.rz(rz_params[base + q], q)

    pairs = _resolve_entanglement(n, entanglement)

    for r in range(reps):
        _ry_layer(r)
        if insert_barriers:
            qc.barrier()
        for (i, j) in pairs:
            qc.cx(i, j)
        if insert_barriers:
            qc.barrier()
        if add_rz_after_entanglement:
            _rz_layer(r)
            if insert_barriers:
                qc.barrier()

    if not skip_final_rotation_layer:
        _ry_layer(reps)

    qc.input_params = params  # type: ignore[attr-defined]
    qc.num_rot_layers = num_rot_layers  # type: ignore[attr-defined]
    if rz_params is not None:
        qc.rz_input_params = rz_params  # type: ignore[attr-defined]
    return qc


class RealAmplitudesExtended(QuantumCircuit):
    """Class‑style wrapper for the extended RealAmplitudes ansatz."""

    def __init__(
        self,
        num_qubits: int,
        reps: int = 1,
        entanglement: EntanglementSpec = "full",
        skip_final_rotation_layer: bool = False,
        insert_barriers: bool = False,
        parameter_prefix: str = "theta",
        rz_parameter_prefix: str = "phi",
        add_rz_after_entanglement: bool = False,
        name: str = "RealAmplitudesExtended",
    ) -> None:
        built = real_amplitudes_extended(
            num_qubits=num_qubits,
            reps=reps,
            entanglement=entanglement,
            skip_final_rotation_layer=skip_final_rotation_layer,
            insert_barriers=insert_barriers,
            parameter_prefix=parameter_prefix,
            rz_parameter_prefix=rz_parameter_prefix,
            add_rz_after_entanglement=add_rz_after_entanglement,
            name=name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)

        self.input_params = built.input_params  # type: ignore[attr-defined]
        self.num_rot_layers = built.num_rot_layers  # type: ignore[attr-defined]
        if hasattr(built, "rz_input_params"):
            self.rz_input_params = built.rz_input_params  # type: ignore[attr-defined]


__all__ = ["RealAmplitudesExtended", "real_amplitudes_extended"]
