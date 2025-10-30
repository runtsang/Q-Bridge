"""RealAmplitudes ansatz with optional symmetry enforcement.

The original RealAmplitudes circuit applies a distinct RY rotation to each qubit
followed by a fixed entanglement pattern.  This module introduces a `symmetry`
flag that, when set, forces all qubits in a given rotation layer to share the
same rotation angle.  This reduces the parameter count and can be useful
in variational algorithms where global rotations are sufficient or where
parameter sharing is desired for regularization purposes.

The interface mirrors the original builder but with an additional keyword
argument.  The `input_params` attribute reflects the actual number of
parameters exposed: it is a `ParameterVector` of length
`num_rot_layers * n` for the unsymmetrised case and of length
`num_rot_layers` when symmetry is enforced.

The class wrapper `RealAmplitudesSymmetry` behaves like Qiskit's
`QuantumCircuit` subclass, exposing the same convenience methods and
attributes as the original `RealAmplitudes`.

Example
-------
>>> from real_amplitudes_controlled_modification import real_amplitudes_symmetric
>>> qc = real_amplitudes_symmetric(num_qubits=3, reps=2, symmetry=True)
>>> print(qc)
3 qubits, 0 classical bits

The circuit contains two rotation layers with a single shared parameter each
followed by the default full‑entanglement CX pattern.

"""

from __future__ import annotations

from typing import Callable, List, Sequence, Tuple

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector


def _resolve_entanglement(
    num_qubits: int,
    entanglement: str
    | Sequence[Tuple[int, int]]
    | Callable[[int], Sequence[Tuple[int, int]]],
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


def real_amplitudes_symmetric(
    num_qubits: int,
    reps: int = 1,
    entanglement: str
    | Sequence[Tuple[int, int]]
    | Callable[[int], Sequence[Tuple[int, int]]] = "full",
    skip_final_rotation_layer: bool = False,
    insert_barriers: bool = False,
    symmetry: bool = False,
    parameter_prefix: str = "theta",
    name: str | None = None,
) -> QuantumCircuit:
    """
    Build a RealAmplitudes-style ``QuantumCircuit`` with optional symmetry.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    reps : int, default 1
        Number of RY/CX repetition blocks.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]]
        Entanglement pattern between qubits.
    skip_final_rotation_layer : bool, default False
        If True, omit the final rotation layer after the last CX block.
    insert_barriers : bool, default False
        If True, insert barriers between layers for easier debugging.
    symmetry : bool, default False
        When True, all qubits in a rotation layer share the same rotation angle.
    parameter_prefix : str, default "theta"
        Prefix for the generated parameter names.
    name : str | None, default None
        Name of the resulting circuit.

    Returns
    -------
    QuantumCircuit
        The constructed ansatz circuit.

    Notes
    -----
    * If *symmetry* is ``False`` the circuit behaves like the original
      ``RealAmplitudes`` implementation.
    * When *symmetry* is ``True`` the number of parameters is reduced to
      ``num_rot_layers``.  The same parameter is applied to every qubit in
      each rotation layer.
    * The ``input_params`` attribute reflects the actual number of parameters
      exposed.  It can be bound to a parameter vector of the same length.
    """
    if num_qubits < 1:
        raise ValueError("num_qubits must be >= 1.")

    n = int(num_qubits)
    qc = QuantumCircuit(n, name=name or "RealAmplitudesSymmetry")

    num_rot_layers = reps if skip_final_rotation_layer else reps + 1

    # Create parameter vector(s) based on symmetry flag
    if symmetry:
        # One parameter per rotation layer
        params = ParameterVector(parameter_prefix, num_rot_layers)
    else:
        # One parameter per qubit per layer
        params = ParameterVector(parameter_prefix, num_rot_layers * n)

    def _rotation_layer(layer_idx: int) -> None:
        """Apply RY rotations for a single layer."""
        if symmetry:
            # All qubits share the same parameter for this layer
            for q in range(n):
                qc.ry(params[layer_idx], q)
        else:
            base = layer_idx * n
            for q in range(n):
                qc.ry(params[base + q], q)

    pairs = _resolve_entanglement(n, entanglement)

    for r in range(reps):
        _rotation_layer(r)
        if insert_barriers:
            qc.barrier()
        for (i, j) in pairs:
            qc.cx(i, j)
        if insert_barriers:
            qc.barrier()

    if not skip_final_rotation_layer:
        _rotation_layer(reps)

    qc.input_params = params  # type: ignore[attr-defined]
    qc.num_rot_layers = num_rot_layers  # type: ignore[attr-defined]
    return qc


class RealAmplitudesSymmetry(QuantumCircuit):
    """Class-style wrapper for the symmetric RealAmplitudes ansatz.

    Parameters
    ----------
    num_qubits : int
        Number of qubits.
    reps : int, default 1
        Number of repetition blocks.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]]
        Entanglement pattern.
    skip_final_rotation_layer : bool, default False
        Whether to omit the final rotation layer.
    insert_barriers : bool, default False
        Whether to insert barriers between layers.
    symmetry : bool, default False
        Whether to enforce symmetry across qubits in each rotation layer.
    parameter_prefix : str, default "theta"
        Prefix for parameter names.
    name : str, default "RealAmplitudesSymmetry"
        Name of the circuit.

    The class mirrors the original ``RealAmplitudes`` wrapper but adds the
    *symmetry* keyword.  The wrapped circuit is composed into the subclass
    instance, and the ``input_params`` and ``num_rot_layers`` attributes are
    forwarded from the builder.
    """

    def __init__(
        self,
        num_qubits: int,
        reps: int = 1,
        entanglement: str
        | Sequence[Tuple[int, int]]
        | Callable[[int], Sequence[Tuple[int, int]]] = "full",
        skip_final_rotation_layer: bool = False,
        insert_barriers: bool = False,
        symmetry: bool = False,
        parameter_prefix: str = "theta",
        name: str = "RealAmplitudesSymmetry",
    ) -> None:
        built = real_amplitudes_symmetric(
            num_qubits=num_qubits,
            reps=reps,
            entanglement=entanglement,
            skip_final_rotation_layer=skip_final_rotation_layer,
            insert_barriers=insert_barriers,
            symmetry=symmetry,
            parameter_prefix=parameter_prefix,
            name=name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)

        self.input_params = built.input_params  # type: ignore[attr-defined]
        self.num_rot_layers = built.num_rot_layers  # type: ignore[attr-defined]


__all__ = ["RealAmplitudesSymmetry", "real_amplitudes_symmetric"]
