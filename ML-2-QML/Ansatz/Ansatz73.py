"""SymmetricRealAmplitudes ansatz builder (parameter‑sharing variant of RealAmplitudes)."""

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
    share_params_across_qubits: bool = False,
    share_params_across_layers: bool = False,
    name: str | None = None,
) -> QuantumCircuit:
    """
    Build a parameter‑sharing variant of the RealAmplitudes ansatz.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    reps : int, default 1
        Number of entanglement + rotation layers.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]], default "full"
        Entanglement pattern.  See :func:`_resolve_entanglement` for accepted values.
    skip_final_rotation_layer : bool, default False
        When ``True`` the last rotation layer is omitted, matching the behaviour of the
        original RealAmplitudes builder.
    insert_barriers : bool, default False
        Insert barriers between layers for clearer visualisation.
    parameter_prefix : str, default "theta"
        Prefix for the parameter vector.
    share_params_across_qubits : bool, default False
        If ``True``, all qubits in a given rotation layer share the same rotation parameter.
    share_params_across_layers : bool, default False
        If ``True``, all layers use the same rotation parameter, overriding
        ``share_params_across_qubits``.
    name : str | None, default None
        Name of the resulting circuit.

    Returns
    -------
    QuantumCircuit
        A circuit implementing the symmetric RealAmplitudes ansatz.
    """
    if num_qubits < 1:
        raise ValueError("num_qubits must be >= 1.")
    if reps < 1:
        raise ValueError("reps must be >= 1.")

    n = int(num_qubits)
    qc = QuantumCircuit(n, name=name or "SymmetricRealAmplitudes")

    effective_layers = reps if skip_final_rotation_layer else reps + 1

    # Determine the number of independent parameters
    if share_params_across_layers:
        param_len = 1
    elif share_params_across_qubits:
        param_len = effective_layers
    else:
        param_len = effective_layers * n

    params = ParameterVector(parameter_prefix, param_len)

    def _rotation_layer(layer_idx: int) -> None:
        """Apply the rotation layer at ``layer_idx``."""
        if share_params_across_layers:
            param = params[0]
            for q in range(n):
                qc.ry(param, q)
        elif share_params_across_qubits:
            param = params[layer_idx]
            for q in range(n):
                qc.ry(param, q)
        else:
            base = layer_idx * n
            for q in range(n):
                qc.ry(params[base + q], q)

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

    qc.input_params = params
    qc.num_rot_layers = effective_layers
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
        parameter_prefix: str = "theta",
        share_params_across_qubits: bool = False,
        share_params_across_layers: bool = False,
        name: str = "SymmetricRealAmplitudes",
    ) -> None:
        built = symmetric_real_amplitudes(
            num_qubits=num_qubits,
            reps=reps,
            entanglement=entanglement,
            skip_final_rotation_layer=skip_final_rotation_layer,
            insert_barriers=insert_barriers,
            parameter_prefix=parameter_prefix,
            share_params_across_qubits=share_params_across_qubits,
            share_params_across_layers=share_params_across_layers,
            name=name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)
        self.input_params = built.input_params
        self.num_rot_layers = built.num_rot_layers


__all__ = ["SymmetricRealAmplitudes", "symmetric_real_amplitudes"]
