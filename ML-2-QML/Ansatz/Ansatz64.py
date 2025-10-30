"""Shared-parameter RealAmplitudes ansatz (controlled‑modification variant)."""

from __future__ import annotations

from typing import Callable, Iterable, List, Sequence, Tuple

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector


# --------------------------------------------------------------------------- #
# Entanglement helper
# --------------------------------------------------------------------------- #
def _resolve_entanglement(
    num_qubits: int,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]],
) -> List[Tuple[int, int]]:
    """Return a list of two‑qubit pairs according to a simple entanglement spec.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]]
        Entanglement specification.  ``"full"``, ``"linear"``, or ``"circular"``
        are accepted as shortcuts.  Otherwise a sequence of pairs or a callable
        that returns such a sequence is expected.

    Returns
    -------
    List[Tuple[int, int]]
        A validated list of qubit pairs for CX gates.
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

    pairs = [(int(i), int(j)) for (i, j) in entanglement]  # type: ignore[arg-type]
    for (i, j) in pairs:
        if i == j:
            raise ValueError("Entanglement pairs must connect distinct qubits.")
        if not (0 <= i < num_qubits and 0 <= j < num_qubits):
            raise ValueError(f"Entanglement pair {(i, j)} out of range for n={num_qubits}.")
    return pairs


# --------------------------------------------------------------------------- #
# Ansatz builder
# --------------------------------------------------------------------------- #
def real_amplitudes(
    num_qubits: int,
    reps: int = 1,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
    skip_final_rotation_layer: bool = False,
    insert_barriers: bool = False,
    parameter_prefix: str = "theta",
    name: str | None = None,
) -> QuantumCircuit:
    """
    Build a RealAmplitudes‑style circuit with shared rotation parameters.

    The key modification over the canonical RealAmplitudes ansatz is that
    all rotation layers share the same set of parameters.  This reduces the
    total number of free parameters from ``reps * n`` to simply ``n``,
    imposing a global reflection symmetry across layers.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    reps : int, default 1
        Number of entanglement‑plus‑rotation blocks.  The final rotation
        layer is optional and controlled by ``skip_final_rotation_layer``.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]]
        Entanglement connectivity.  See :func:`_resolve_entanglement` for details.
    skip_final_rotation_layer : bool, default False
        If ``True``, omit the rotation layer after the last entanglement block.
    insert_barriers : bool, default False
        Insert barriers between layers for easier circuit inspection and
        debugging.  Barriers do not affect the unitary.
    parameter_prefix : str, default "theta"
        Prefix for the shared parameter names.
    name : str | None, default None
        Name of the resulting circuit.  If ``None``, defaults to ``"RealAmplitudes"``.

    Returns
    -------
    QuantumCircuit
        The constructed circuit.  The attributes ``input_params`` and
        ``num_rot_layers`` are attached for convenience.
    """
    if num_qubits < 1:
        raise ValueError("num_qubits must be >= 1.")
    if reps < 1:
        raise ValueError("reps must be >= 1.")

    n = int(num_qubits)
    qc = QuantumCircuit(n, name=name or "RealAmplitudes")

    # One shared parameter per qubit
    params = ParameterVector(parameter_prefix, n)

    def _rotation_layer() -> None:
        for q in range(n):
            qc.ry(params[q], q)

    pairs = _resolve_entanglement(n, entanglement)

    for _ in range(reps):
        _rotation_layer()
        if insert_barriers:
            qc.barrier()
        for (i, j) in pairs:
            qc.cx(i, j)
        if insert_barriers:
            qc.barrier()

    if not skip_final_rotation_layer:
        _rotation_layer()

    qc.input_params = params  # type: ignore[attr-defined]
    qc.num_rot_layers = reps + (0 if skip_final_rotation_layer else 1)  # type: ignore[attr-defined]
    return qc


# --------------------------------------------------------------------------- #
# Class wrapper
# --------------------------------------------------------------------------- #
class RealAmplitudes(QuantumCircuit):
    """Class‑style wrapper for the shared‑parameter RealAmplitudes ansatz.

    The class behaves identically to :func:`real_amplitudes` but exposes the
    resulting circuit as a subclass of :class:`qiskit.circuit.QuantumCircuit`,
    allowing seamless integration into Qiskit workflows.
    """
    def __init__(
        self,
        num_qubits: int,
        reps: int = 1,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
        skip_final_rotation_layer: bool = False,
        insert_barriers: bool = False,
        parameter_prefix: str = "theta",
        name: str = "RealAmplitudes",
    ) -> None:
        built = real_amplitudes(
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


__all__ = ["real_amplitudes", "RealAmplitudes"]
