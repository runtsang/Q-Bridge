"""RealAmplitudesControlled ansatz with parameter sharing and symmetry options."""
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
    share_params: bool = False,
    enforce_symmetry: bool = False,
    name: str | None = None,
) -> QuantumCircuit:
    """
    Build a RealAmplitudes‑style circuit with optional parameter sharing and qubit‑pair symmetry.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    reps : int, default 1
        Number of entanglement + rotation blocks.
    entanglement : str or sequence or callable, default "full"
        Specification of two‑qubit entanglement pattern.
    skip_final_rotation_layer : bool, default False
        If True, omit the rotation layer after the last entanglement block.
    insert_barriers : bool, default False
        Insert a barrier after each rotation and entanglement block.
    parameter_prefix : str, default "theta"
        Prefix for parameter names.
    share_params : bool, default False
        If True, use a single ParameterVector for all rotation layers.
    enforce_symmetry : bool, default False
        If True, enforce θ_i = θ_{n-1-i} within each rotation layer.
    name : str, optional
        Circuit name.

    Returns
    -------
    QuantumCircuit
        The constructed ansatz.
    """
    if num_qubits < 1:
        raise ValueError("num_qubits must be >= 1.")

    n = int(num_qubits)
    qc = QuantumCircuit(n, name=name or "RealAmplitudesControlled")

    num_rot_layers = reps if skip_final_rotation_layer else reps + 1

    # Parameter handling
    if share_params:
        params_layers = [ParameterVector(parameter_prefix, n)]
    else:
        params_layers = [
            ParameterVector(f"{parameter_prefix}_l{layer}", n) for layer in range(num_rot_layers)
        ]

    def _rotation_layer(layer_idx: int) -> None:
        params = params_layers[layer_idx]
        for q in range(n):
            # Symmetry handling: map qubit i to its mirror j = n-1-i
            if enforce_symmetry:
                mirror = n - 1 - q
                if q < mirror:
                    # use parameter from the mirror qubit
                    param = params[mirror]
                else:
                    param = params[q]
            else:
                param = params[q]
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

    # Expose metadata
    qc.input_params = params_layers  # type: ignore[attr-defined]
    qc.num_rot_layers = num_rot_layers  # type: ignore[attr-defined]
    return qc


class RealAmplitudesControlled(QuantumCircuit):
    """
    Class‑style wrapper for the RealAmplitudesControlled ansatz.

    Parameters
    ----------
    num_qubits : int
        Number of qubits.
    reps : int, default 1
        Number of entanglement + rotation blocks.
    entanglement : str or sequence or callable, default "full"
        Entanglement pattern.
    skip_final_rotation_layer : bool, default False
        Skip the last rotation layer.
    insert_barriers : bool, default False
        Insert barriers between blocks.
    parameter_prefix : str, default "theta"
        Prefix for parameters.
    share_params : bool, default False
        Share parameters across layers.
    enforce_symmetry : bool, default False
        Enforce symmetry between qubit pairs.
    name : str, default "RealAmplitudesControlled"
        Circuit name.
    """

    def __init__(
        self,
        num_qubits: int,
        reps: int = 1,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
        skip_final_rotation_layer: bool = False,
        insert_barriers: bool = False,
        parameter_prefix: str = "theta",
        share_params: bool = False,
        enforce_symmetry: bool = False,
        name: str = "RealAmplitudesControlled",
    ) -> None:
        built = real_amplitudes_controlled(
            num_qubits=num_qubits,
            reps=reps,
            entanglement=entanglement,
            skip_final_rotation_layer=skip_final_rotation_layer,
            insert_barriers=insert_barriers,
            parameter_prefix=parameter_prefix,
            share_params=share_params,
            enforce_symmetry=enforce_symmetry,
            name=name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)

        self.input_params = built.input_params  # type: ignore[attr-defined]
        self.num_rot_layers = built.num_rot_layers  # type: ignore[attr-defined]


__all__ = ["RealAmplitudesControlled", "real_amplitudes_controlled"]
