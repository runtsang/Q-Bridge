"""RealAmplitudesSymmetry ansatz (parameter sharing across qubits)."""

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


def real_amplitudes_symmetry(
    num_qubits: int,
    reps: int = 1,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
    skip_final_rotation_layer: bool = False,
    insert_barriers: bool = False,
    parameter_prefix: str = "theta",
    name: str | None = None,
) -> QuantumCircuit:
    """
    Build a RealAmplitudes‑style ``QuantumCircuit`` with *parameter sharing* across qubits.

    This controlled modification enforces that all qubits in a given rotation layer
    receive the same RY rotation angle.  Consequently, the number of parameters per
    layer is reduced from ``num_qubits`` to ``1``.  The entanglement pattern and
    overall layering remain identical to the original RealAmplitudes ansatz.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit. Must be >= 1.
    reps : int, default 1
        Number of entanglement layers. The number of rotation layers is
        ``reps`` if ``skip_final_rotation_layer`` is True, otherwise ``reps + 1``.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]]
        Specification of the entanglement pattern. Accepted strings are
        ``"full"``, ``"linear"``, and ``"circular"``.  Alternatively, a sequence
        of qubit pairs or a callable returning such a sequence may be provided.
    skip_final_rotation_layer : bool, default False
        If True, omit the rotation layer after the last entanglement block.
    insert_barriers : bool, default False
        If True, insert barriers between logical blocks for clarity.
    parameter_prefix : str, default "theta"
        Prefix for the generated parameter names.
    name : str | None, default None
        Optional name for the circuit.  Defaults to "RealAmplitudesSymmetry".

    Returns
    -------
    QuantumCircuit
        The constructed parameterized circuit.

    Raises
    ------
    ValueError
        If ``num_qubits`` < 1 or if the entanglement specification is invalid.
    """
    if num_qubits < 1:
        raise ValueError("num_qubits must be >= 1.")

    n = int(num_qubits)
    qc = QuantumCircuit(n, name=name or "RealAmplitudesSymmetry")

    num_rot_layers = reps if skip_final_rotation_layer else reps + 1
    # One parameter per rotation layer (shared across all qubits)
    params = ParameterVector(parameter_prefix, num_rot_layers)

    def _rotation_layer(layer_idx: int) -> None:
        """Apply a shared RY rotation to all qubits for the given layer."""
        param = params[layer_idx]
        for q in range(n):
            qc.ry(param, q)

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
    """Class‑style wrapper that behaves like Qiskit's ``RealAmplitudesSymmetry``."""

    def __init__(
        self,
        num_qubits: int,
        reps: int = 1,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
        skip_final_rotation_layer: bool = False,
        insert_barriers: bool = False,
        parameter_prefix: str = "theta",
        name: str = "RealAmplitudesSymmetry",
    ) -> None:
        built = real_amplitudes_symmetry(
            num_qubits=num_qubits,
            reps=reps,
            entanglement=entanglement,
            skip_final_rotation_layer=skip_final_rotation_layer,
            insert_barriers=insert_barriers,
            parameter_prefix=parameter_prefix,
            name=name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)

        self.input_params = built.input_params  # type: ignore[attr-defined]
        self.num_rot_layers = built.num_rot_layers  # type: ignore[attr-defined]


__all__ = ["RealAmplitudesSymmetry", "real_amplitudes_symmetry"]
