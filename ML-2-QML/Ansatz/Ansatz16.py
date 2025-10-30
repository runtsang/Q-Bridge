"""RealAmplitudesControlled ansatz builder (RY + CX layers with optional parameter sharing and reversible entanglement)."""
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


def real_amplitudes_controlled(
    num_qubits: int,
    reps: int = 1,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
    skip_final_rotation_layer: bool = False,
    insert_barriers: bool = False,
    parameter_sharing: bool = False,
    reverse_entanglement: bool = False,
    parameter_prefix: str = "theta",
    name: str | None = None,
) -> QuantumCircuit:
    """
    Build a RealAmplitudes‑style ``QuantumCircuit`` with optional symmetry controls.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    reps : int, default 1
        Number of rotation–entangler repetitions.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]]
        Entanglement specification.  Accepted strings are ``"full"``, ``"linear"``,
        and ``"circular"``.  Alternatively provide a custom sequence of qubit pairs
        or a callable that returns such a sequence for a given ``num_qubits``.
    skip_final_rotation_layer : bool, default False
        If ``True`` omit the rotation layer that normally follows the last
        entanglement block.  This reduces the total number of parameters.
    insert_barriers : bool, default False
        Insert barriers between layers for easier circuit inspection.
    parameter_sharing : bool, default False
        When ``True`` all rotation layers share the same set of parameters
        (one parameter per qubit).  This enforces a global symmetry across
        layers.
    reverse_entanglement : bool, default False
        When ``True`` an additional entanglement block using the reverse
        order of the original pairs is applied after each forward
        entanglement.  The resulting pattern is symmetric under qubit
        reversal.
    parameter_prefix : str, default "theta"
        Prefix for automatically generated parameter names.
    name : str, optional
        Name of the resulting quantum circuit.

    Returns
    -------
    QuantumCircuit
        The constructed ansatz circuit.  The circuit exposes two attributes:
        ``input_params`` – the :class:`~qiskit.circuit.ParameterVector` (or
        list of them when ``parameter_sharing`` is ``True``) – and
        ``num_rot_layers`` – the number of RY rotation layers that will be
        applied when the circuit is executed.

    Notes
    -----
    - The ansatz preserves the original RY + CX structure of
      ``RealAmplitudes`` but adds the optional symmetry controls.
    - ``parameter_sharing`` and ``reverse_entanglement`` are mutually
      independent; they can be combined for more expressive circuits.
    """
    if num_qubits < 1:
        raise ValueError("num_qubits must be >= 1.")
    if reps < 0:
        raise ValueError("reps must be >= 0.")

    n = int(num_qubits)
    qc = QuantumCircuit(n, name=name or "RealAmplitudesControlled")

    # Determine how many rotation layers will actually be applied
    num_rot_layers = reps if skip_final_rotation_layer else reps + 1

    # Parameter handling
    flat_params: ParameterVector | None = None
    if parameter_sharing:
        # One shared ParameterVector for all layers
        shared_params = ParameterVector(parameter_prefix, n)
        param_objs = [shared_params] * num_rot_layers
        flat_params = shared_params
    else:
        # Independent parameters per layer
        total_params = num_rot_layers * n
        flat_params = ParameterVector(parameter_prefix, total_params)
        param_objs = [
            flat_params[layer_idx * n : (layer_idx + 1) * n] for layer_idx in range(num_rot_layers)
        ]

    def _rotation_layer(layer_idx: int) -> None:
        """Apply an RY rotation layer using the parameters of the given layer."""
        params_layer = param_objs[layer_idx]
        for q in range(n):
            qc.ry(params_layer[q], q)

    pairs = _resolve_entanglement(n, entanglement)

    for r in range(reps):
        _rotation_layer(r)
        if insert_barriers:
            qc.barrier()
        # Forward entanglement
        for (i, j) in pairs:
            qc.cx(i, j)
        # Optional reverse entanglement
        if reverse_entanglement:
            for (i, j) in reversed(pairs):
                qc.cx(i, j)
        if insert_barriers:
            qc.barrier()

    if not skip_final_rotation_layer:
        _rotation_layer(reps)

    # Attach metadata
    qc.input_params = flat_params  # type: ignore[attr-defined]
    qc.num_rot_layers = num_rot_layers  # type: ignore[attr-defined]
    return qc


class RealAmplitudesControlled(QuantumCircuit):
    """Class‑style wrapper that behaves like Qiskit's ``RealAmplitudesControlled``."""

    def __init__(
        self,
        num_qubits: int,
        reps: int = 1,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
        skip_final_rotation_layer: bool = False,
        insert_barriers: bool = False,
        parameter_sharing: bool = False,
        reverse_entanglement: bool = False,
        parameter_prefix: str = "theta",
        name: str = "RealAmplitudesControlled",
    ) -> None:
        built = real_amplitudes_controlled(
            num_qubits=num_qubits,
            reps=reps,
            entanglement=entanglement,
            skip_final_rotation_layer=skip_final_rotation_layer,
            insert_barriers=insert_barriers,
            parameter_sharing=parameter_sharing,
            reverse_entanglement=reverse_entanglement,
            parameter_prefix=parameter_prefix,
            name=name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)

        self.input_params = built.input_params  # type: ignore[attr-defined]
        self.num_rot_layers = built.num_rot_layers  # type: ignore[attr-defined]


__all__ = ["RealAmplitudesControlled", "real_amplitudes_controlled"]
