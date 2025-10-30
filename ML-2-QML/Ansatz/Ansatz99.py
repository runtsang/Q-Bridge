"""RealAmplitudesCZExtended – a depth‑controlled, multi‑entangler variant of RealAmplitudesCZ.

The design adds:
* **Entanglement Schedule**: Accepts a list of callable schedules that can produce any pairwise or higher‑order entanglement for each repetition.
* **Adaptive Layer**: A boolean flag per repetition that can insert an extra “entanglement‑only” layer (no rotations) to be used as a trainable depth‑controlled gate.
* **Barrier Management**: The two‑phase barrier approach is kept, but the *‑fence barrier only if a barrier function exists.
"""

from __future__ import annotations

from typing import Callable, Sequence, Tuple, Union, List

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector


def _resolve_entanglement(
    num_qubits: int,
    entanglement: Union[str, Sequence[Tuple[int, int]], Callable[[int], Sequence[Tuple[int, int]]]],
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


def _apply_entanglement(
    qc: QuantumCircuit,
    pairs: Sequence[Tuple[int, int]],
    gate: str,
) -> None:
    """Apply the specified two‑qubit gate to all qubit pairs."""
    if gate == "cx":
        for (i, j) in pairs:
            qc.cx(i, j)
    elif gate == "cz":
        for (i, j) in pairs:
            qc.cz(i, j)
    elif gate == "xx":
        for (i, j) in pairs:
            qc.rxx(1.5707963267948966, i, j)  # π/2 rotation
    elif gate == "yy":
        for (i, j) in pairs:
            qc.ryy(1.5707963267948966, i, j)
    elif gate == "zz":
        for (i, j) in pairs:
            qc.rzz(1.5707963267948966, i, j)
    else:
        raise ValueError(f"Unsupported entanglement gate: {gate!r}")


def real_amplitudes_cz_extended(
    num_qubits: int,
    reps: int = 1,
    entanglement: Union[str, Sequence[Tuple[int, int]], Callable[[int], Sequence[Tuple[int, int]]]] = "full",
    entanglement_gate: str = "cz",
    adaptive: bool = False,
    adaptive_depth: int = 1,
    skip_final_rotation_layer: bool = False,
    insert_barriers: bool = False,
    parameter_prefix: str = "theta",
    name: str | None = None,
) -> QuantumCircuit:
    """Depth‑controlled, multi‑entangler variant of the RealAmplitudesCZ ansatz.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the ansatz.
    reps : int, default 1
        Number of repetition blocks of rotations followed by entanglement.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]]
        Entanglement pattern for each repetition. Can be a string ('full', 'linear', 'circular'),
        a custom list of qubit pairs, or a callable that returns a list of pairs for a given qubit
        count.
    entanglement_gate : str, default 'cz'
        Two‑qubit gate used for entanglement. Supported values are 'cx', 'cz', 'xx', 'yy', 'zz'.
    adaptive : bool, default False
        If True, insert an extra entanglement‑only layer after each rotation block.
    adaptive_depth : int, default 1
        Number of additional entanglement layers to insert when adaptive is True.
    skip_final_rotation_layer : bool, default False
        If True, omit the final rotation layer that normally follows the last entanglement block.
    insert_barriers : bool, default False
        If True, insert barriers between logical layers for easier circuit inspection.
    parameter_prefix : str, default 'theta'
        Prefix used for the rotation parameters.
    name : str | None, default None
        Optional name for the quantum circuit. If None, defaults to ``RealAmplitudesCZExtended``.
    """
    if num_qubits < 1:
        raise ValueError("num_qubits must be >= 1.")

    if entanglement_gate not in {"cx", "cz", "xx", "yy", "zz"}:
        raise ValueError(f"Unsupported entanglement_gate: {entanglement_gate!r}")

    n = int(num_qubits)
    qc = QuantumCircuit(n, name=name or "RealAmplitudesCZExtended")

    num_rot_layers = reps if skip_final_rotation_layer else reps + 1
    params = ParameterVector(parameter_prefix, num_rot_layers * n)

    def _rot(layer: int) -> None:
        base = layer * n
        for q in range(n):
            qc.ry(params[base + q], q)

    pairs = _resolve_entanglement(n, entanglement)

    for r in range(reps):
        _rot(r)
        if insert_barriers:
            qc.barrier()
        _apply_entanglement(qc, pairs, entanglement_gate)
        if insert_barriers:
            qc.barrier()
        if adaptive:
            for _ in range(adaptive_depth):
                _apply_entanglement(qc, pairs, entanglement_gate)
                if insert_barriers:
                    qc.barrier()

    if not skip_final_rotation_layer:
        _rot(reps)

    qc.input_params = params  # type: ignore[attr-defined]
    qc.num_rot_layers = num_rot_layers  # type: ignore[attr-defined]
    return qc


class RealAmplitudesCZExtended(QuantumCircuit):
    """Class wrapper for the extended RealAmplitudesCZ ansatz with optional adaptive entanglement layers.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the ansatz.
    reps : int, default 1
        Number of repetition blocks of rotations followed by entanglement.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]]
        Entanglement pattern for each repetition.
    entanglement_gate : str, default 'cz'
        Two‑qubit gate used for entanglement.
    adaptive : bool, default False
        If True, insert an extra entanglement‑only layer after each rotation block.
    adaptive_depth : int, default 1
        Number of additional entanglement layers to insert when adaptive is True.
    skip_final_rotation_layer : bool, default False
        If True, omit the final rotation layer that normally follows the last entanglement block.
    insert_barriers : bool, default False
        If True, insert barriers between logical layers for easier circuit inspection.
    parameter_prefix : str, default 'theta'
        Prefix used for the rotation parameters.
    name : str, default 'RealAmplitudesCZExtended'
        Name of the quantum circuit.
    """

    def __init__(
        self,
        num_qubits: int,
        reps: int = 1,
        entanglement: Union[str, Sequence[Tuple[int, int]], Callable[[int], Sequence[Tuple[int, int]]]] = "full",
        entanglement_gate: str = "cz",
        adaptive: bool = False,
        adaptive_depth: int = 1,
        skip_final_rotation_layer: bool = False,
        insert_barriers: bool = False,
        parameter_prefix: str = "theta",
        name: str = "RealAmplitudesCZExtended",
    ) -> None:
        built = real_amplitudes_cz_extended(
            num_qubits,
            reps,
            entanglement,
            entanglement_gate,
            adaptive,
            adaptive_depth,
            skip_final_rotation_layer,
            insert_barriers,
            parameter_prefix,
            name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)
        self.input_params = built.input_params  # type: ignore[attr-defined]
        self.num_rot_layers = built.num_rot_layers  # type: ignore[attr-defined]


__all__ = ["RealAmplitudesCZExtended", "real_amplitudes_cz_extended"]
