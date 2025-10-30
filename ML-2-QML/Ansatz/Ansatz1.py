"""RealAmplitudesControlled ansatz builder (shared-parameter RY layers with CX entanglers)."""

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
    parameter_prefix: str = "theta",
    name: str | None = None,
) -> QuantumCircuit:
    """Build a symmetry‑constrained RealAmplitudes-style circuit.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    reps : int, default 1
        Number of entanglement + rotation blocks.
    entanglement : str or sequence or callable, default "full"
        Entanglement schedule. Accepted strings are ``"full"``, ``"linear"``,
        and ``"circular"``. Custom schedules can also be supplied.
    skip_final_rotation_layer : bool, default False
        If ``True``, the circuit ends with an entanglement block
        and does not apply the final rotation layer.
    insert_barriers : bool, default False
        Insert barriers between blocks for visual clarity.
    parameter_prefix : str, default "theta"
        Prefix for the parameter vector names.
    name : str, optional
        Circuit name.

    Returns
    -------
    QuantumCircuit
        A parameterised circuit with shared RY rotations per layer.
    """
    if num_qubits < 1:
        raise ValueError("num_qubits must be >= 1.")

    n = int(num_qubits)
    qc = QuantumCircuit(n, name=name or "RealAmplitudesControlled")

    num_rot_layers = reps if skip_final_rotation_layer else reps + 1
    params = ParameterVector(parameter_prefix, num_rot_layers)

    def _rotation_layer(layer_idx: int) -> None:
        """Apply a rotation layer where all qubits share the same angle."""
        for q in range(n):
            qc.ry(params[layer_idx], q)

    pairs = _resolve_entanglement(n, entanglement)

    for r in range(reps):
        # Entanglement first – reversed ordering from the original ansatz
        for (i, j) in pairs:
            qc.cx(i, j)
        if insert_barriers:
            qc.barrier()
        _rotation_layer(r)
        if insert_barriers:
            qc.barrier()

    if not skip_final_rotation_layer:
        # Final rotation layer without preceding entanglement
        _rotation_layer(reps)

    qc.input_params = params  # type: ignore[attr-defined]
    qc.num_rot_layers = num_rot_layers  # type: ignore[attr-defined]
    return qc


class RealAmplitudesControlled(QuantumCircuit):
    """Class‑style wrapper for the symmetry‑constrained RealAmplitudes ansatz."""

    def __init__(
        self,
        num_qubits: int,
        reps: int = 1,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
        skip_final_rotation_layer: bool = False,
        insert_barriers: bool = False,
        parameter_prefix: str = "theta",
        name: str = "RealAmplitudesControlled",
    ) -> None:
        built = real_amplitudes_controlled(
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


__all__ = ["RealAmplitudesControlled", "real_amplitudes_controlled"]
