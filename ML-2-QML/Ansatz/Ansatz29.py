"""
RealAmplitudes CZ variant with extended entanglement and depth control.

This module defines a new ansatz ``RealAmplitudesCZExtended`` that augments the
original ``RealAmplitudesCZ`` by:
  * Adding a ``depth`` parameter that allows the circuit to be built to a
    customized depth (the number of repetition blocks).
  * **Hybrid entangler**: each repetition alternates between a CZ gate and
    an iSWAP gate when ``entanglement_gate='hybrid'``; otherwise a single
    entanglement gate type is used for all layers.
  * **Parameter sharing**: optional sharing of rotation angles across
    all layers (``parameter_sharing=True``).
  * **Barriers**: configurable barrier placement.
  * **Qiskit‑compatible**: exposes both a convenience function and a
    subclass that inherits from ``QuantumCircuit``.
"""

from __future__ import annotations

from typing import Callable, List, Sequence, Tuple

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector


def _resolve_entanglement(
    num_qubits: int,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]],
) -> List[Tuple[int, int]]:
    """Return a list of two‑qubit pairs according to a simple entanglement spec.

    Supports the same string specifications as the original ``RealAmplitudesCZ``
    ("full", "linear", "circular") and allows custom sequences or callables.
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

    # Sequence of pairs
    pairs = [(int(i), int(j)) for (i, j) in entanglement]  # type: ignore[arg-type]
    for (i, j) in pairs:
        if i == j:
            raise ValueError("Entanglement pairs must connect distinct qubits.")
        if not (0 <= i < num_qubits and 0 <= j < num_qubits):
            raise ValueError(f"Entanglement pair {(i, j)} out of range for n={num_qubits}.")
    return pairs


def real_amplitudes_cz_extended(
    num_qubits: int,
    reps: int = 1,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
    entanglement_gate: str = "cz",
    skip_final_rotation_layer: bool = False,
    insert_barriers: bool = False,
    parameter_sharing: bool = False,
    parameter_prefix: str = "theta",
    name: str | None = None,
) -> QuantumCircuit:
    """Build an extended RealAmplitudes CZ ansatz.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    reps : int, default 1
        Number of repetition blocks.  Controls the depth.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]]
        Specification of which qubit pairs to entangle.  Supports the same
        options as the original ansatz.
    entanglement_gate : str, default "cz"
        Gate to use for entanglement.  One of ``"cz"``, ``"iswap"``, ``"swap"``
        or ``"hybrid"``.  If ``"hybrid"``, each repetition alternates between
        CZ and iSWAP.
    skip_final_rotation_layer : bool, default False
        If True, omit the final rotation layer.
    insert_barriers : bool, default False
        If True, insert a barrier before and after each entanglement block.
    parameter_sharing : bool, default False
        If True, all rotation layers share the same set of parameters.
    parameter_prefix : str, default "theta"
        Prefix for the parameter names.
    name : str | None, default None
        Name of the circuit.

    Returns
    -------
    QuantumCircuit
        The constructed ansatz circuit.
    """
    if num_qubits < 1:
        raise ValueError("num_qubits must be >= 1.")
    if reps < 1:
        raise ValueError("reps must be >= 1.")
    if entanglement_gate not in {"cz", "iswap", "swap", "hybrid"}:
        raise ValueError(f"Unsupported entanglement_gate: {entanglement_gate!r}")

    n = int(num_qubits)
    qc = QuantumCircuit(n, name=name or "RealAmplitudesCZExtended")

    num_rot_layers = reps if skip_final_rotation_layer else reps + 1
    if parameter_sharing:
        params = ParameterVector(parameter_prefix, n)
    else:
        params = ParameterVector(parameter_prefix, num_rot_layers * n)

    def _rot(layer: int) -> None:
        if parameter_sharing:
            for q in range(n):
                qc.ry(params[q], q)
        else:
            base = layer * n
            for q in range(n):
                qc.ry(params[base + q], q)

    pairs = _resolve_entanglement(n, entanglement)

    for r in range(reps):
        _rot(r)
        if insert_barriers:
            qc.barrier()
        for (i, j) in pairs:
            if entanglement_gate == "cz":
                qc.cz(i, j)
            elif entanglement_gate == "iswap":
                qc.iswap(i, j)
            elif entanglement_gate == "swap":
                qc.swap(i, j)
            else:  # hybrid
                if r % 2 == 0:
                    qc.cz(i, j)
                else:
                    qc.iswap(i, j)
        if insert_barriers:
            qc.barrier()

    if not skip_final_rotation_layer:
        _rot(reps)

    qc.input_params = params  # type: ignore[attr-defined]
    qc.num_rot_layers = num_rot_layers  # type: ignore[attr-defined]
    qc.entanglement_gate = entanglement_gate  # type: ignore[attr-defined]
    qc.parameter_sharing = parameter_sharing  # type: ignore[attr-defined]
    return qc


class RealAmplitudesCZExtended(QuantumCircuit):
    """Convenience class wrapper for the extended RealAmplitudes CZ ansatz."""

    def __init__(
        self,
        num_qubits: int,
        reps: int = 1,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
        entanglement_gate: str = "cz",
        skip_final_rotation_layer: bool = False,
        insert_barriers: bool = False,
        parameter_sharing: bool = False,
        parameter_prefix: str = "theta",
        name: str = "RealAmplitudesCZExtended",
    ) -> None:
        built = real_amplitudes_cz_extended(
            num_qubits,
            reps,
            entanglement,
            entanglement_gate,
            skip_final_rotation_layer,
            insert_barriers,
            parameter_sharing,
            parameter_prefix,
            name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)
        self.input_params = built.input_params  # type: ignore[attr-defined]
        self.num_rot_layers = built.num_rot_layers  # type: ignore[attr-defined]
        self.entanglement_gate = built.entanglement_gate  # type: ignore[attr-defined]
        self.parameter_sharing = built.parameter_sharing  # type: ignore[attr-defined]


__all__ = ["RealAmplitudesCZExtended", "real_amplitudes_cz_extended"]
