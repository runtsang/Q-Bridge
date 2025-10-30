"""RealAmplitudes ansatz with optional parameter sharing.

This module extends the canonical RealAmplitudes construction by adding a
`shared_parameters` flag. When enabled, all rotation layers reuse the same
parameter vector, effectively enforcing a global symmetry and cutting the
number of free parameters from `n * num_rot_layers` to `n`. The entanglement
pattern and layer ordering remain unchanged, so the ansatz is still
recognisable as a RealAmplitudes circuit.

Typical usage:
    qc = RealAmplitudes(num_qubits=4, reps=3, shared_parameters=True)
    qc.draw()
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


def real_amplitudes(
    num_qubits: int,
    reps: int = 1,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
    skip_final_rotation_layer: bool = False,
    insert_barriers: bool = False,
    parameter_prefix: str = "theta",
    shared_parameters: bool = False,
    name: str | None = None,
) -> QuantumCircuit:
    """Build a RealAmplitudes‑style circuit with optional parameter sharing.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    reps : int, default 1
        Number of rotation‑entanglement repetitions.
    entanglement : str or sequence or callable, default "full"
        Specification of two‑qubit entanglement pattern.
    skip_final_rotation_layer : bool, default False
        If True, omit the rotation layer after the last entanglement.
    insert_barriers : bool, default False
        Whether to insert barriers between layers for readability.
    parameter_prefix : str, default "theta"
        Prefix for the rotation parameters.
    shared_parameters : bool, default False
        If True, all rotation layers use the same `ParameterVector`,
        reducing the total number of free parameters.
    name : str | None, default None
        Name of the circuit.

    Returns
    -------
    QuantumCircuit
        The constructed circuit with attributes ``input_params`` and
        ``num_rot_layers`` for introspection.
    """
    if num_qubits < 1:
        raise ValueError("num_qubits must be >= 1.")

    n = int(num_qubits)
    qc = QuantumCircuit(n, name=name or "RealAmplitudes")

    # Determine the number of rotation layers (including the optional final one)
    num_rot_layers = reps if skip_final_rotation_layer else reps + 1

    # Allocate parameters
    if shared_parameters:
        # One parameter per qubit reused across all layers
        params = ParameterVector(parameter_prefix, n)
    else:
        # Full set of parameters: one per qubit per layer
        params = ParameterVector(parameter_prefix, num_rot_layers * n)

    def _rotation_layer(layer_idx: int | None = None) -> None:
        """Apply a rotation layer.

        When ``shared_parameters`` is True, ``layer_idx`` is ignored and the
        same parameters are used for every rotation.
        """
        if shared_parameters:
            base = 0
        else:
            base = layer_idx * n  # type: ignore[arg-type]
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


class RealAmplitudes(QuantumCircuit):
    """Convenience wrapper for the RealAmplitudes ansatz with optional parameter sharing.

    The class mirrors Qiskit's ``RealAmplitudes`` but exposes the ``shared_parameters``
    keyword.  All other arguments are forwarded to :func:`real_amplitudes`.

    Attributes
    ----------
    input_params : ParameterVector
        The parameters used by the circuit.
    num_rot_layers : int
        The number of rotation layers (including the optional final layer).
    """

    def __init__(
        self,
        num_qubits: int,
        reps: int = 1,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
        skip_final_rotation_layer: bool = False,
        insert_barriers: bool = False,
        parameter_prefix: str = "theta",
        shared_parameters: bool = False,
        name: str = "RealAmplitudes",
    ) -> None:
        built = real_amplitudes(
            num_qubits=num_qubits,
            reps=reps,
            entanglement=entanglement,
            skip_final_rotation_layer=skip_final_rotation_layer,
            insert_barriers=insert_barriers,
            parameter_prefix=parameter_prefix,
            shared_parameters=shared_parameters,
            name=name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)

        self.input_params = built.input_params  # type: ignore[attr-defined]
        self.num_rot_layers = built.num_rot_layers  # type: ignore[attr-defined]


__all__ = ["RealAmplitudes", "real_amplitudes"]
