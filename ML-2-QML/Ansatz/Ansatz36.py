"""RealAmplitudesCZExtended: depth‑controlled hybrid entanglement with optional global phase."""
from __future__ import annotations

from typing import Callable, Iterable, List, Sequence, Tuple

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterVector

# --------------------------------------------------------------------------- #
# Helper: Entanglement pair resolution
# --------------------------------------------------------------------------- #
def _resolve_entanglement(
    num_qubits: int,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]],
) -> List[Tuple[int, int]]:
    """Return a list of two‑qubit pairs according to a simple entanglement spec.
    The function accepts the same interface as the original seed and performs
    validation, raising informative errors for invalid pairs.
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
# Main ansatz construction
# --------------------------------------------------------------------------- #
def real_amplitudes_cz_extended(
    num_qubits: int,
    reps: int = 1,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
    hybrid_entanglement: str = "CZ",
    global_phase: bool = False,
    skip_final_rotation_layer: bool = False,
    insert_barriers: bool = False,
    parameter_prefix: str = "theta",
    name: str | None = None,
) -> QuantumCircuit:
    """
    A depth‑controlled hybrid entanglement variant of the RealAmplitudesCZ ansatz.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    reps : int, default 1
        Number of rotation‑entanglement repetitions.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]]
        Entanglement schedule.  Accepted strings are ``"full"``, ``"linear"``, and
        ``"circular"``.  Any custom sequence or callable returning a sequence of
        pairs is also supported.
    hybrid_entanglement : str, default "CZ"
        Type of two‑qubit entangler to use.  ``"CZ"`` uses a CZ gate, ``"CRZ"`` uses a
        controlled‑RZ gate with a fixed angle of ``π/2``, and ``"alternate"``
        alternates between CZ and CRZ across repetitions.
    global_phase : bool, default False
        If ``True``, a global RZ rotation is inserted after each entanglement
        layer, providing an additional tunable degree of freedom.
    skip_final_rotation_layer : bool, default False
        If ``True``, the final rotation layer after the last entanglement is omitted.
    insert_barriers : bool, default False
        If ``True``, barriers are inserted between rotation, entanglement, and
        global‑phase blocks for clearer visual separation.
    parameter_prefix : str, default "theta"
        Prefix used for all parameter vectors.
    name : str | None, default None
        Optional name for the constructed circuit.

    Returns
    -------
    QuantumCircuit
        The constructed parameterised ansatz.

    Notes
    -----
    * The rotation layers use a single parameter per qubit.
    * If ``global_phase`` is enabled, an additional parameter per qubit per
      entanglement repetition is added.
    * The circuit exposes ``input_params`` (a :class:`ParameterVector` containing
      all parameters in order) and ``num_rot_layers`` (the number of rotation
      layers, including the optional final layer).
    """
    if num_qubits < 1:
        raise ValueError("num_qubits must be >= 1.")
    if reps < 1:
        raise ValueError("reps must be >= 1.")
    if hybrid_entanglement not in {"CZ", "CRZ", "alternate"}:
        raise ValueError(f"hybrid_entanglement must be one of 'CZ', 'CRZ', or 'alternate', got {hybrid_entanglement!r}.")

    n = int(num_qubits)
    qc = QuantumCircuit(n, name=name or "RealAmplitudesCZExtended")

    # Determine the number of rotation layers
    num_rot_layers = reps if skip_final_rotation_layer else reps + 1

    # Parameter allocation
    total_rot_params = num_rot_layers * n
    total_phase_params = n * reps if global_phase else 0
    total_params = total_rot_params + total_phase_params

    param_all = ParameterVector(f"{parameter_prefix}_all", total_params)
    rot_params = param_all[:total_rot_params]
    phase_params = param_all[total_rot_params:] if global_phase else None

    def _rot(layer: int) -> None:
        base = layer * n
        for q in range(n):
            qc.ry(rot_params[base + q], q)

    def _global_phase_layer(layer: int) -> None:
        base = layer * n
        for q in range(n):
            qc.rz(phase_params[base + q], q)

    pairs = _resolve_entanglement(n, entanglement)

    for r in range(reps):
        _rot(r)
        if insert_barriers:
            qc.barrier()
        for (i, j) in pairs:
            if hybrid_entanglement == "CZ":
                qc.cz(i, j)
            elif hybrid_entanglement == "CRZ":
                qc.crz(np.pi / 2, i, j)
            else:  # alternate
                if r % 2 == 0:
                    qc.cz(i, j)
                else:
                    qc.crz(np.pi / 2, i, j)
        if global_phase:
            _global_phase_layer(r)
        if insert_barriers:
            qc.barrier()

    if not skip_final_rotation_layer:
        _rot(reps)

    qc.input_params = param_all  # type: ignore[attr-defined]
    qc.num_rot_layers = num_rot_layers  # type: ignore[attr-defined]
    return qc


# --------------------------------------------------------------------------- #
# Class wrapper
# --------------------------------------------------------------------------- #
class RealAmplitudesCZExtended(QuantumCircuit):
    """Convenience subclass wrapping :func:`real_amplitudes_cz_extended`."""
    def __init__(
        self,
        num_qubits: int,
        reps: int = 1,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
        hybrid_entanglement: str = "CZ",
        global_phase: bool = False,
        skip_final_rotation_layer: bool = False,
        insert_barriers: bool = False,
        parameter_prefix: str = "theta",
        name: str = "RealAmplitudesCZExtended",
    ) -> None:
        built = real_amplitudes_cz_extended(
            num_qubits,
            reps,
            entanglement,
            hybrid_entanglement,
            global_phase,
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
