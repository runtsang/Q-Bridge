"""SymmetricRealAmplitudes ansatz (controlled modification of RealAmplitudes)."""

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
    symmetric: bool = True,
    shared_params: bool = False,
    reverse_entanglement: bool = False,
    parameter_prefix: str = "theta",
    name: str | None = None,
) -> QuantumCircuit:
    """
    Build a RealAmplitudes‑style ansatz with optional rotational symmetry and
    parameter sharing across layers.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    reps : int, default 1
        Number of RY+CX repetition blocks.  If ``skip_final_rotation_layer`` is
        ``False`` an additional rotation layer is appended.
    entanglement : str, sequence or callable, default "full"
        Entanglement pattern for the CX gates.
    skip_final_rotation_layer : bool, default False
        If ``True`` the final rotation layer is omitted.
    insert_barriers : bool, default False
        Insert a barrier before and after each entangling block.
    symmetric : bool, default True
        Enforce ``theta_q = theta_{n-1-q}`` for all rotation layers.
    shared_params : bool, default False
        Use the same set of rotation parameters for every layer.
    reverse_entanglement : bool, default False
        Reverse the order of entanglement pairs on alternating layers.
    parameter_prefix : str, default "theta"
        Prefix for the parameter vector.
    name : str, optional
        Name of the resulting circuit.

    Returns
    -------
    QuantumCircuit
        The constructed ansatz circuit.

    Notes
    -----
    * When ``symmetric`` is ``True`` only ``ceil(num_qubits/2)`` distinct
      rotation parameters are needed per layer.
    * If ``shared_params`` is ``True`` the same parameters are reused across all
      layers, further reducing the parameter count.
    * The function performs input validation and raises informative errors
      for invalid arguments.
    """
    if num_qubits < 1:
        raise ValueError("num_qubits must be >= 1.")
    if reps < 0:
        raise ValueError("reps must be >= 0.")
    if not isinstance(symmetric, bool):
        raise TypeError("symmetric must be a bool.")
    if not isinstance(shared_params, bool):
        raise TypeError("shared_params must be a bool.")
    if not isinstance(reverse_entanglement, bool):
        raise TypeError("reverse_entanglement must be a bool.")

    n = int(num_qubits)
    qc = QuantumCircuit(n, name=name or "SymmetricRealAmplitudes")

    # Determine number of rotation layers
    num_rot_layers = reps if skip_final_rotation_layer else reps + 1

    # Parameter vector length
    half = (n + 1) // 2  # ceil(n/2)
    if shared_params:
        param_count = half
    else:
        param_count = num_rot_layers * half
    params = ParameterVector(parameter_prefix, param_count)

    # Helper to map a (layer, qubit) pair to a parameter
    def _param(layer: int, qubit: int) -> ParameterVector:
        sym_idx = min(qubit, n - 1 - qubit)  # enforce symmetry
        if shared_params:
            return params[sym_idx]
        base = layer * half
        return params[base + sym_idx]

    # Build layers
    pairs = _resolve_entanglement(n, entanglement)
    for r in range(reps):
        # Rotation layer
        for q in range(n):
            qc.ry(_param(r, q), q)
        if insert_barriers:
            qc.barrier()
        # Entanglement layer
        ent_pairs = list(reversed(pairs)) if reverse_entanglement and r % 2 == 1 else pairs
        for (i, j) in ent_pairs:
            qc.cx(i, j)
        if insert_barriers:
            qc.barrier()

    if not skip_final_rotation_layer:
        for q in range(n):
            qc.ry(_param(reps, q), q)

    # Attach metadata
    qc.input_params = params  # type: ignore[attr-defined]
    qc.num_rot_layers = num_rot_layers  # type: ignore[attr-defined]
    qc.symmetric = symmetric  # type: ignore[attr-defined]
    qc.shared_params = shared_params  # type: ignore[attr-defined]
    return qc


class SymmetricRealAmplitudes(QuantumCircuit):
    """Class‑style wrapper for the SymmetricRealAmplitudes ansatz."""

    def __init__(
        self,
        num_qubits: int,
        reps: int = 1,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
        skip_final_rotation_layer: bool = False,
        insert_barriers: bool = False,
        symmetric: bool = True,
        shared_params: bool = False,
        reverse_entanglement: bool = False,
        parameter_prefix: str = "theta",
        name: str = "SymmetricRealAmplitudes",
    ) -> None:
        built = symmetric_real_amplitudes(
            num_qubits=num_qubits,
            reps=reps,
            entanglement=entanglement,
            skip_final_rotation_layer=skip_final_rotation_layer,
            insert_barriers=insert_barriers,
            symmetric=symmetric,
            shared_params=shared_params,
            reverse_entanglement=reverse_entanglement,
            parameter_prefix=parameter_prefix,
            name=name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)

        self.input_params = built.input_params  # type: ignore[attr-defined]
        self.num_rot_layers = built.num_rot_layers  # type: ignore[attr-defined]
        self.symmetric = built.symmetric  # type: ignore[attr-defined]
        self.shared_params = built.shared_params  # type: ignore[attr-defined]


__all__ = ["SymmetricRealAmplitudes", "symmetric_real_amplitudes"]
