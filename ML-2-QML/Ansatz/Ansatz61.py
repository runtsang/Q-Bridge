"""SymmetricRealAmplitudes: a symmetry‑constrained real‑amplitudes ansatz."""
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


def symmetric_real_amplitudes(
    num_qubits: int,
    reps: int = 1,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
    skip_final_rotation_layer: bool = False,
    insert_barriers: bool = False,
    parameter_prefix: str = "theta",
    name: str | None = None,
) -> QuantumCircuit:
    """Build a symmetry‑constrained RealAmplitudes‑style circuit.

    The rotation parameters are mirrored about the central qubit(s), which
    halves the number of free parameters per layer.  All other features
    (entanglement pattern, depth, optional barriers) mirror the original
    ``RealAmplitudes`` construction.

    Parameters
    ----------
    num_qubits
        Number of qubits in the circuit.
    reps
        Number of entangling layers.
    entanglement
        Specification of two‑qubit entanglement as in the original ansatz.
    skip_final_rotation_layer
        If ``True``, no final rotation layer is added after the last entangling
        block.
    insert_barriers
        Insert barriers between layers for easier visual inspection.
    parameter_prefix
        Prefix for the symbolic parameters.
    name
        Optional name for the circuit.

    Returns
    -------
    QuantumCircuit
        The constructed symmetric ansatz.
    """
    if num_qubits < 1:
        raise ValueError("num_qubits must be >= 1.")
    n = int(num_qubits)
    qc = QuantumCircuit(n, name=name or "SymmetricRealAmplitudes")

    # Number of symmetry‑constrained parameters per layer
    n_sym = (n + 1) // 2
    num_rot_layers = reps if skip_final_rotation_layer else reps + 1
    params = ParameterVector(parameter_prefix, num_rot_layers * n_sym)

    def _rotation_layer(layer_idx: int) -> None:
        base = layer_idx * n_sym
        for q in range(n):
            sym_idx = min(q, n - 1 - q)
            qc.ry(params[base + sym_idx], q)

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


class SymmetricRealAmplitudes(QuantumCircuit):
    """Convenience class wrapper that mimics Qiskit's ``RealAmplitudes``.

    The constructor builds a symmetry‑constrained ansatz and composes it into
    the instance.  All attributes from the underlying circuit are exposed,
    including ``input_params`` and ``num_rot_layers``.
    """

    def __init__(
        self,
        num_qubits: int,
        reps: int = 1,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
        skip_final_rotation_layer: bool = False,
        insert_barriers: bool = False,
        parameter_prefix: str = "theta",
        name: str = "SymmetricRealAmplitudes",
    ) -> None:
        built = symmetric_real_amplitudes(
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


__all__ = ["SymmetricRealAmplitudes", "symmetric_real_amplitudes"]
