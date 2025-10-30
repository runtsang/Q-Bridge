"""RealAmplitudes variant with alternating RY/RX rotations and optional symmetry constraints.

This module implements a controlled modification of the standard
`RealAmplitudesAlternating` ansatz.  The new ansatz adds two optional
modifiers:

* ``share_params_across_layers`` – If ``True`` the same rotation parameters
  are reused for every repetition, greatly reducing the number of free
  parameters.
* ``mirror_symmetry`` – If ``True`` the rotation parameters are constrained
  to be symmetric with respect to the qubit index (``theta[i] = theta[n-1-i]``).
  This enforces a reflection symmetry on the circuit.

These knobs allow the user to trade expressibility for a smaller parameter
space or to impose physical symmetries on the variational circuit.

The public API mirrors the original implementation: a convenience function
``real_amplitudes_alternating_controlled_modification`` and a subclass
``RealAmplitudesAlternatingControlledModification`` of :class:`QuantumCircuit`.
"""

from __future__ import annotations

from typing import Callable, Iterable, List, Sequence, Tuple

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector


def _resolve_entanglement(
    num_qubits: int,
    spec: str | Sequence[Tuple[int, int]] | Callable[[int], Iterable[Tuple[int, int]]],
) -> List[Tuple[int, int]]:
    """Return a list of two‑qubit pairs according to a compact specification.

    Parameters
    ----------
    num_qubits
        Number of qubits in the circuit.
    spec
        Either a string describing a standard pattern, an explicit list of
        pairs, or a callable that receives *num_qubits* and returns an
        iterable of pairs.

    Returns
    -------
    List[Tuple[int, int]]
        Validated list of qubit pairs.

    Raises
    ------
    ValueError
        If *spec* is a string that is not recognised, or if any pair contains
        an invalid qubit index.
    """
    if isinstance(spec, str):
        if spec == "full":
            return [(i, j) for i in range(num_qubits) for j in range(i + 1, num_qubits)]
        if spec == "linear":
            return [(i, i + 1) for i in range(num_qubits - 1)]
        if spec == "circular":
            pairs = [(i, i + 1) for i in range(num_qubits - 1)]
            if num_qubits > 2:
                pairs.append((num_qubits - 1, 0))
            return pairs
        raise ValueError(f"Unknown entanglement string: {spec!r}")

    if callable(spec):
        pairs = list(spec(num_qubits))
    else:
        pairs = list(spec)

    validated: List[Tuple[int, int]] = []
    for i, j in pairs:
        if i == j:
            raise ValueError("Entanglement pairs must connect distinct qubits.")
        if not (0 <= i < num_qubits and 0 <= j < num_qubits):
            raise ValueError(f"Entanglement pair {(i, j)} out of range for n={num_qubits}.")
        validated.append((int(i), int(j)))
    return validated


def real_amplitudes_alternating_controlled_modification(
    num_qubits: int,
    reps: int = 1,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Iterable[Tuple[int, int]]] = "full",
    skip_final_rotation_layer: bool = False,
    insert_barriers: bool = False,
    parameter_prefix: str = "theta",
    share_params_across_layers: bool = False,
    mirror_symmetry: bool = False,
    name: str | None = None,
) -> QuantumCircuit:
    """Build a RealAmplitudes with alternating rotations and optional symmetry.

    Parameters
    ----------
    num_qubits
        Number of qubits in the ansatz.
    reps
        Number of entanglement layers.
    entanglement
        Specification of the two‑qubit entanglement pattern.
    skip_final_rotation_layer
        Whether to omit the last rotation layer that would normally follow
        the final entanglement operation.
    insert_barriers
        Insert barriers between logical sub‑blocks for easier circuit
        inspection.
    parameter_prefix
        Prefix used for all rotation parameters.
    share_params_across_layers
        Reuse the same set of rotation parameters for every repetition.
    mirror_symmetry
        Enforce reflection symmetry on rotation parameters across the qubit
        indices.
    name
        Optional name for the resulting circuit.

    Returns
    -------
    QuantumCircuit
        The constructed ansatz circuit.

    Raises
    ------
    ValueError
        If *num_qubits* is less than 1.
    """
    if num_qubits < 1:
        raise ValueError("num_qubits must be >= 1.")

    n = int(num_qubits)
    qc = QuantumCircuit(n, name=name or "RealAmplitudesAlternatingControlledModification")

    # Determine the number of rotation layers that will actually be
    # instantiated in the circuit.
    num_rot_layers = reps if skip_final_rotation_layer else reps + 1

    # Number of distinct parameters per layer depending on symmetry.
    params_per_layer = (n + 1) // 2 if mirror_symmetry else n

    # Total number of parameters, possibly shared across layers.
    total_params = params_per_layer if share_params_across_layers else params_per_layer * num_rot_layers

    params = ParameterVector(parameter_prefix, total_params)

    # Helper to compute the parameter index for a given layer and qubit.
    def _param_index(layer: int, qubit: int) -> int:
        if share_params_across_layers:
            layer = 0
        # Mirror symmetry: pair (q, n-1-q) share the same parameter.
        idx = min(qubit, n - 1 - qubit) if mirror_symmetry else qubit
        return layer * params_per_layer + idx

    # Resolve entanglement pairs once.
    pairs = _resolve_entanglement(n, entanglement)

    for r in range(reps):
        # Rotation layer
        for q in range(n):
            idx = _param_index(r, q)
            if r % 2 == 0:
                qc.ry(params[idx], q)
            else:
                qc.rx(params[idx], q)
        if insert_barriers:
            qc.barrier()
        # Entanglement layer
        for i, j in pairs:
            qc.cx(i, j)
        if insert_barriers:
            qc.barrier()

    # Final rotation layer, if requested.
    if not skip_final_rotation_layer:
        for q in range(n):
            idx = _param_index(reps, q)
            if reps % 2 == 0:
                qc.ry(params[idx], q)
            else:
                qc.rx(params[idx], q)

    qc.input_params = params  # type: ignore[attr-defined]
    qc.num_rot_layers = num_rot_layers  # type: ignore[attr-defined]
    return qc


class RealAmplitudesAlternatingControlledModification(QuantumCircuit):
    """Convenience wrapper around :func:`real_amplitudes_alternating_controlled_modification`."""

    def __init__(
        self,
        num_qubits: int,
        reps: int = 1,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Iterable[Tuple[int, int]]] = "full",
        skip_final_rotation_layer: bool = False,
        insert_barriers: bool = False,
        parameter_prefix: str = "theta",
        share_params_across_layers: bool = False,
        mirror_symmetry: bool = False,
        name: str = "RealAmplitudesAlternatingControlledModification",
    ) -> None:
        built = real_amplitudes_alternating_controlled_modification(
            num_qubits,
            reps,
            entanglement,
            skip_final_rotation_layer,
            insert_barriers,
            parameter_prefix,
            share_params_across_layers,
            mirror_symmetry,
            name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)
        self.input_params = built.input_params  # type: ignore[attr-defined]
        self.num_rot_layers = built.num_rot_layers  # type: ignore[attr-defined]


__all__ = [
    "real_amplitudes_alternating_controlled_modification",
    "RealAmplitudesAlternatingControlledModification",
]
