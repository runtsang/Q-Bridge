"""ControlledRealAmplitudes ansatz (mirrored RY rotations with optional alternating entanglement)."""
from __future__ import annotations

from typing import Callable, Iterable, List, Sequence, Tuple

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterVector


def _resolve_entanglement(
    num_qubits: int,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]],
) -> List[Tuple[int, int]]:
    """Return a list of two‑qubit pairs according to a simple entanglement spec.

    Supports the original ``full``, ``linear`` and ``circular`` strings as well as a new
    ``alternating`` schedule that pairs qubits as ``(0,1)``, ``(2,3)``, … and optionally
    wraps the last pair to the first qubit.
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
        if entanglement == "alternating":
            pairs: List[Tuple[int, int]] = []
            for i in range(0, num_qubits - 1, 2):
                pairs.append((i, i + 1))
            # Wrap the last pair to the first qubit if odd number of qubits
            if num_qubits % 2 == 1:
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


def controlled_real_amplitudes(
    num_qubits: int,
    reps: int = 1,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
    skip_final_rotation_layer: bool = False,
    insert_barriers: bool = False,
    symmetry: bool = True,
    parameter_prefix: str = "theta",
    name: str | None = None,
) -> QuantumCircuit:
    """
    Build a symmetry‑constrained RealAmplitudes circuit.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    reps : int, default 1
        Number of RY‑CX repetition blocks.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]]
        Entanglement pattern. Supports ``"full"``, ``"linear"``, ``"circular"``,
        ``"alternating"``, a sequence of pairs, or a callable that generates pairs.
    skip_final_rotation_layer : bool, default False
        If True, omit the final rotation layer that follows the last entangling block.
    insert_barriers : bool, default False
        Insert a barrier after each rotation or entangling block for readability.
    symmetry : bool, default True
        When True, rotation angles are mirrored across the qubit axis
        (theta_q == theta_{n-1-q} for each layer).
    parameter_prefix : str, default "theta"
        Prefix for the parameters in the ParameterVector.
    name : str | None, default None
        Optional name for the resulting QuantumCircuit.

    Returns
    -------
    QuantumCircuit
        A parameterised circuit with mirrored RY rotations and the chosen entanglement.
    """
    if num_qubits < 1:
        raise ValueError("num_qubits must be >= 1.")
    if reps < 1:
        raise ValueError("reps must be >= 1.")

    n = int(num_qubits)
    qc = QuantumCircuit(n, name=name or "ControlledRealAmplitudes")

    # Number of rotation layers (including optional final layer)
    num_rot_layers = reps if skip_final_rotation_layer else reps + 1

    # Number of independent parameters per layer
    half_qubits = (n + 1) // 2  # ceil(n/2)
    base_params = ParameterVector(parameter_prefix, num_rot_layers * half_qubits)

    # Helper to apply a single rotation layer using symmetry
    def _rotation_layer(layer_idx: int) -> None:
        base = layer_idx * half_qubits
        for q in range(n):
            param_idx = base + min(q, n - 1 - q)
            qc.ry(base_params[param_idx], q)

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

    # Expose both the full parameter set (mirrored) and the underlying independent set
    qc.input_params = base_params  # type: ignore[attr-defined]
    qc.base_params = base_params  # type: ignore[attr-defined]
    qc.num_rot_layers = num_rot_layers  # type: ignore[attr-defined]
    qc.symmetry = symmetry  # type: ignore[attr-defined]
    return qc


class ControlledRealAmplitudes(QuantumCircuit):
    """Class‑style wrapper for ControlledRealAmplitudes.

    The constructor mirrors the functional interface and exposes the same
    ``input_params`` and ``base_params`` attributes for parameter binding.
    """

    def __init__(
        self,
        num_qubits: int,
        reps: int = 1,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
        skip_final_rotation_layer: bool = False,
        insert_barriers: bool = False,
        symmetry: bool = True,
        parameter_prefix: str = "theta",
        name: str = "ControlledRealAmplitudes",
    ) -> None:
        built = controlled_real_amplitudes(
            num_qubits=num_qubits,
            reps=reps,
            entanglement=entanglement,
            skip_final_rotation_layer=skip_final_rotation_layer,
            insert_barriers=insert_barriers,
            symmetry=symmetry,
            parameter_prefix=parameter_prefix,
            name=name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)

        self.input_params = built.input_params  # type: ignore[attr-defined]
        self.base_params = built.base_params  # type: ignore[attr-defined]
        self.num_rot_layers = built.num_rot_layers  # type: ignore[attr-defined]
        self.symmetry = built.symmetry  # type: ignore[attr-defined]


__all__ = ["ControlledRealAmplitudes", "controlled_real_amplitudes"]
