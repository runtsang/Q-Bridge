"""RealAmplitudesSymmetric ansatz builder (RY rotation layers with CX entanglers and reflection symmetry)."""
from __future__ import annotations

from typing import Callable, List, Sequence, Tuple

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector


def _resolve_entanglement(
    num_qubits: int,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]],
) -> List[Tuple[int, int]]:
    """Return a list of two‑qubit pairs according to a simple entanglement spec.

    This helper is identical to the one in the original ``real_amplitudes`` ansatz and
    accepts the same ``entanglement`` specification.  It validates input and raises
    informative errors when an invalid pair is supplied.
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


def real_amplitudes_symmetric(
    num_qubits: int,
    reps: int = 1,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
    skip_final_rotation_layer: bool = False,
    insert_barriers: bool = False,
    parameter_prefix: str = "theta",
    name: str | None = None,
) -> QuantumCircuit:
    """Build a RealAmplitudes-style quantum circuit with reflection symmetry across qubits.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    reps : int, default 1
        Number of repetition blocks (rotation + entanglement).
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]]
        Entanglement pattern. See ``_resolve_entanglement`` for accepted values.
    skip_final_rotation_layer : bool, default False
        If ``True``, omit the last rotation layer that normally follows the final entanglement.
    insert_barriers : bool, default False
        If ``True``, insert a barrier after each rotation and entanglement block.
    parameter_prefix : str, default "theta"
        Prefix for the parameters in the ``ParameterVector``.
    name : str | None, default None
        Optional circuit name. Defaults to ``"RealAmplitudesSymmetric"``.

    Notes
    -----
    The ansatz imposes a reflection symmetry: for each rotation layer the angles satisfy
    ``theta_i = theta_{n-1-i}``.  Consequently, the number of independent parameters is
    ``ceil(n/2) * num_rot_layers``.  This symmetry can be useful for problems with
    left-right or particle-hole invariance.

    Returns
    -------
    QuantumCircuit
        The constructed circuit with ``input_params`` and ``num_rot_layers`` attributes.
    """
    if num_qubits < 1:
        raise ValueError("num_qubits must be >= 1.")

    n = int(num_qubits)
    qc = QuantumCircuit(n, name=name or "RealAmplitudesSymmetric")

    num_rot_layers = reps if skip_final_rotation_layer else reps + 1
    half = (n + 1) // 2  # ceil(n/2)
    params = ParameterVector(parameter_prefix, half * num_rot_layers)

    def _rotation_layer(layer_idx: int) -> None:
        base = layer_idx * half
        for q in range(n):
            # Map qubit index to the symmetric parameter index
            sym_idx = q if q < half else n - 1 - q
            qc.ry(params[base + sym_idx], q)

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

    qc.input_params = params  # type: ignore[attr-defined]
    qc.num_rot_layers = num_rot_layers  # type: ignore[attr-defined]
    return qc


class RealAmplitudesSymmetric(QuantumCircuit):
    """Class-style wrapper for the symmetric RealAmplitudes ansatz."""

    def __init__(
        self,
        num_qubits: int,
        reps: int = 1,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
        skip_final_rotation_layer: bool = False,
        insert_barriers: bool = False,
        parameter_prefix: str = "theta",
        name: str = "RealAmplitudesSymmetric",
    ) -> None:
        built = real_amplitudes_symmetric(
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


__all__ = ["RealAmplitudesSymmetric", "real_amplitudes_symmetric"]
