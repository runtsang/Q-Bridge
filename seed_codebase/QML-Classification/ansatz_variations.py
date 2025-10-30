"""
Standalone, lightweight re-implementation of Qiskit's classic `RealAmplitudes`
**plus variants** in a single-file module. Use this to build Ry–entangler ansätze
(and a couple tweaks) without importing `qiskit.circuit.library.RealAmplitudes`.

Included builders (class + function for each):

• `RealAmplitudes` / `real_amplitudes(...)`
    - Canonical layout: [RY layer] → [CX entanglement] repeated `reps`, then
      an optional final RY layer (on by default).

• `RealAmplitudesAlternating` / `real_amplitudes_alternating(...)`
    - Rotation layers alternate **RY** (even layers) and **RX** (odd layers).

• `RealAmplitudesCZ` / `real_amplitudes_cz(...)`
    - Same as canonical but uses **CZ** for entanglement instead of CX.

All variants depend only on Qiskit core circuit objects (`QuantumCircuit`,
`ParameterVector`). Each circuit exposes `input_params` (a flat `ParameterVector`)
for easy parameter binding and `num_rot_layers` indicating how many rotation
layers exist in the circuit instance.

Example
-------
>>> from real_amplitudes_with_variants import RealAmplitudes, RealAmplitudesAlternating, RealAmplitudesCZ
>>> ans_base = RealAmplitudes(4, reps=2, entanglement="linear")
>>> ans_alt  = RealAmplitudesAlternating(4, reps=2, entanglement="linear")
>>> ans_cz   = RealAmplitudesCZ(4, reps=2, entanglement="linear")

Binding parameters
------------------
>>> vals = [0.1,0.2,0.3,0.4,  0.5,0.6,0.7,0.8,  0.9,1.0,1.1,1.2]  # for 4 qubits, reps=2 -> 3 rot layers
>>> bound = ans_base.assign_parameters(dict(zip(ans_base.input_params, vals)))
"""
from __future__ import annotations

from typing import Callable, List, Sequence, Tuple

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _resolve_entanglement(
    num_qubits: int,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]],
) -> List[Tuple[int, int]]:
    """Return a list of two-qubit pairs according to a simple entanglement spec.

    Supported specs:
      - "full": all-to-all pairs (i < j)
      - "linear": nearest neighbors (0,1), (1,2), ...
      - "circular": linear plus wrap-around (n-1,0) if n > 2
      - explicit list of pairs like [(0, 2), (1, 3)]
      - callable: f(num_qubits) -> sequence of (i, j)
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

    # sequence of pairs
    pairs = [(int(i), int(j)) for (i, j) in entanglement]  # type: ignore[arg-type]
    # basic validation
    for (i, j) in pairs:
        if i == j:
            raise ValueError("Entanglement pairs must connect distinct qubits.")
        if not (0 <= i < num_qubits and 0 <= j < num_qubits):
            raise ValueError(f"Entanglement pair {(i, j)} out of range for n={num_qubits}.")
    return pairs


# ---------------------------------------------------------------------------
# Canonical RealAmplitudes (RY + CX)
# ---------------------------------------------------------------------------

def real_amplitudes(
    num_qubits: int,
    reps: int = 1,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
    skip_final_rotation_layer: bool = False,
    insert_barriers: bool = False,
    parameter_prefix: str = "theta",
    name: str | None = None,
) -> QuantumCircuit:
    """Build a RealAmplitudes-style `QuantumCircuit` (RY + CX layers)."""
    if num_qubits < 1:
        raise ValueError("num_qubits must be >= 1.")

    n = int(num_qubits)
    qc = QuantumCircuit(n, name=name or "RealAmplitudes")

    # number of rotation layers
    num_rot_layers = reps if skip_final_rotation_layer else reps + 1

    # Create a flat ParameterVector with one parameter per (layer, qubit)
    params = ParameterVector(parameter_prefix, num_rot_layers * n)

    def _rotation_layer(layer_idx: int) -> None:
        base = layer_idx * n
        for q in range(n):
            qc.ry(params[base + q], q)

    pairs = _resolve_entanglement(n, entanglement)

    # Build reps of [Rot -> Ent]
    for r in range(reps):
        _rotation_layer(r)
        if insert_barriers:
            qc.barrier()
        for (i, j) in pairs:
            qc.cx(i, j)
        if insert_barriers:
            qc.barrier()

    # Final rotation layer (optional)
    if not skip_final_rotation_layer:
        _rotation_layer(reps)

    # Expose the input symbols for convenience
    qc.input_params = params  # type: ignore[attr-defined]
    qc.num_rot_layers = num_rot_layers  # type: ignore[attr-defined]
    return qc


class RealAmplitudes(QuantumCircuit):
    """Class-style wrapper that behaves like Qiskit's `RealAmplitudes`."""

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

        # Convenience attributes
        self.input_params = built.input_params  # type: ignore[attr-defined]
        self.num_rot_layers = built.num_rot_layers  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Variant 1: Alternating RY / RX rotation layers
# ---------------------------------------------------------------------------

def real_amplitudes_alternating(
    num_qubits: int,
    reps: int = 1,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
    skip_final_rotation_layer: bool = False,
    insert_barriers: bool = False,
    parameter_prefix: str = "theta",
    name: str | None = None,
) -> QuantumCircuit:
    if num_qubits < 1:
        raise ValueError("num_qubits must be >= 1.")

    n = int(num_qubits)
    qc = QuantumCircuit(n, name=name or "RealAmplitudesAlternating")

    num_rot_layers = reps if skip_final_rotation_layer else reps + 1
    params = ParameterVector(parameter_prefix, num_rot_layers * n)

    def _rot(layer: int) -> None:
        base = layer * n
        # even layers: RY, odd layers: RX
        if layer % 2 == 0:
            for q in range(n):
                qc.ry(params[base + q], q)
        else:
            for q in range(n):
                qc.rx(params[base + q], q)

    pairs = _resolve_entanglement(n, entanglement)

    for r in range(reps):
        _rot(r)
        if insert_barriers:
            qc.barrier()
        for (i, j) in pairs:
            qc.cx(i, j)
        if insert_barriers:
            qc.barrier()

    if not skip_final_rotation_layer:
        _rot(reps)

    qc.input_params = params  # type: ignore[attr-defined]
    qc.num_rot_layers = num_rot_layers  # type: ignore[attr-defined]
    return qc


class RealAmplitudesAlternating(QuantumCircuit):
    def __init__(
        self,
        num_qubits: int,
        reps: int = 1,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
        skip_final_rotation_layer: bool = False,
        insert_barriers: bool = False,
        parameter_prefix: str = "theta",
        name: str = "RealAmplitudesAlternating",
    ) -> None:
        built = real_amplitudes_alternating(
            num_qubits, reps, entanglement,
            skip_final_rotation_layer, insert_barriers,
            parameter_prefix, name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)
        self.input_params = built.input_params  # type: ignore[attr-defined]
        self.num_rot_layers = built.num_rot_layers  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Variant 2: Same as canonical but with CZ entanglement
# ---------------------------------------------------------------------------

def real_amplitudes_cz(
    num_qubits: int,
    reps: int = 1,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
    skip_final_rotation_layer: bool = False,
    insert_barriers: bool = False,
    parameter_prefix: str = "theta",
    name: str | None = None,
) -> QuantumCircuit:
    if num_qubits < 1:
        raise ValueError("num_qubits must be >= 1.")

    n = int(num_qubits)
    qc = QuantumCircuit(n, name=name or "RealAmplitudesCZ")

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
        for (i, j) in pairs:
            qc.cz(i, j)
        if insert_barriers:
            qc.barrier()

    if not skip_final_rotation_layer:
        _rot(reps)

    qc.input_params = params  # type: ignore[attr-defined]
    qc.num_rot_layers = num_rot_layers  # type: ignore[attr-defined]
    return qc


class RealAmplitudesCZ(QuantumCircuit):
    def __init__(
        self,
        num_qubits: int,
        reps: int = 1,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
        skip_final_rotation_layer: bool = False,
        insert_barriers: bool = False,
        parameter_prefix: str = "theta",
        name: str = "RealAmplitudesCZ",
    ) -> None:
        built = real_amplitudes_cz(
            num_qubits, reps, entanglement,
            skip_final_rotation_layer, insert_barriers,
            parameter_prefix, name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)
        self.input_params = built.input_params  # type: ignore[attr-defined]
        self.num_rot_layers = built.num_rot_layers  # type: ignore[attr-defined]


__all__ = [
    # canonical
    "RealAmplitudes",
    "real_amplitudes",
    # variants
    "RealAmplitudesAlternating",
    "real_amplitudes_alternating",
    "RealAmplitudesCZ",
    "real_amplitudes_cz",
]
