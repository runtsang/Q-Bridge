"""RealAmplitudesCZExtended: a depth‑controlled, multi‑entangler variant of RealAmplitudesCZ."""
from __future__ import annotations

from typing import Callable, List, Sequence, Tuple

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

__all__ = ["RealAmplitudesCZExtended", "real_amplitudes_cz_extended"]

# --------------------------------------------------------------------------- #
# Helper utilities
# --------------------------------------------------------------------------- #
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

# --------------------------------------------------------------------------- #
# Main ansatz construction
# --------------------------------------------------------------------------- #
def real_amplitudes_cz_extended(
    num_qubits: int,
    reps: int = 1,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
    skip_final_rotation_layer: bool = False,
    insert_barriers: bool = False,
    parameter_prefix: str = "theta",
    name: str | None = None,
    multiqubit_entanglement: str | None = None,
    block_size: int = 3,
    extra_entanglement_layers: int = 0,
    adaptive: bool = False,
) -> QuantumCircuit:
    """Build a depth‑controlled, multi‑entangler variant of RealAmplitudesCZ.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the ansatz.
    reps : int, default 1
        Number of rotation/entanglement repetition cycles.
    entanglement : str or sequence or callable, default 'full'
        Specification of the base two‑qubit entanglement pattern.
    skip_final_rotation_layer : bool, default False
        If True, omit the final rotation layer after the last repetition.
    insert_barriers : bool, default False
        If True, insert barriers around each rotation and entanglement block.
    parameter_prefix : str, default 'theta'
        Prefix for the parameter vector names.
    name : str, optional
        Name for the resulting circuit.
    multiqubit_entanglement : {'chain', 'block'} or None, default None
        Optional additional entanglement pattern applied after each repetition.
        * ``'chain'`` applies CZ gates between consecutive qubits.
        * ``'block'`` applies all‑to‑all CZ gates within blocks of ``block_size``.
    block_size : int, default 3
        Size of each block for the ``'block'`` multiqubit entanglement.
        Ignored if ``multiqubit_entanglement`` is None.
    extra_entanglement_layers : int, default 0
        Number of additional entanglement layers to insert for each repetition.
        Each layer uses the pattern specified by ``multiqubit_entanglement``.
    adaptive : bool, default False
        If True, insert a rotation layer before every extra entanglement layer
        to allow the ansatz to adapt its amplitudes between entanglement steps.

    Returns
    -------
    QuantumCircuit
        The constructed ansatz circuit.  The circuit exposes two convenience
        attributes:
        * ``input_params`` – the ParameterVector used to parameterise the rotation
          gates.
        * ``num_rot_layers`` – total number of rotation layers in the circuit.

    Notes
    -----
    The parameter vector size is:
        (reps + 1 if not skip_final_rotation_layer else reps +
         extra_entanglement_layers * reps if adaptive else 0) * num_qubits
    """
    if num_qubits < 1:
        raise ValueError("num_qubits must be >= 1.")
    if reps < 0:
        raise ValueError("reps must be non‑negative.")
    if extra_entanglement_layers < 0:
        raise ValueError("extra_entanglement_layers must be non‑negative.")
    if multiqubit_entanglement not in (None, "chain", "block"):
        raise ValueError(
            "multiqubit_entanglement must be None, 'chain', or 'block'."
        )
    if multiqubit_entanglement == "block" and block_size < 2:
        raise ValueError("block_size must be at least 2 for block entanglement.")
    n = int(num_qubits)
    qc = QuantumCircuit(n, name=name or "RealAmplitudesCZExtended")

    # Resolve base entanglement pairs once
    base_pairs = _resolve_entanglement(n, entanglement)

    # Compute total number of rotation layers
    base_rot_layers = reps if skip_final_rotation_layer else reps + 1
    extra_rot_layers = extra_entanglement_layers * reps if adaptive else 0
    total_rot_layers = base_rot_layers + extra_rot_layers
    params = ParameterVector(parameter_prefix, total_rot_layers * n)

    def _rot(layer: int) -> None:
        """Apply a single rotation layer using the ``layer``‑th block of parameters."""
        base = layer * n
        for q in range(n):
            qc.ry(params[base + q], q)

    def _apply_multiqubit_entanglement() -> None:
        """Apply the chosen multiqubit entanglement pattern."""
        if multiqubit_entanglement == "chain":
            for i in range(n - 1):
                qc.cz(i, i + 1)
        elif multiqubit_entanglement == "block":
            for start in range(0, n, block_size):
                end = min(start + block_size, n)
                for i in range(start, end):
                    for j in range(i + 1, end):
                        qc.cz(i, j)

    rot_index = 0
    for r in range(reps):
        _rot(rot_index)
        rot_index += 1
        if insert_barriers:
            qc.barrier()
        for (i, j) in base_pairs:
            qc.cz(i, j)
        if insert_barriers:
            qc.barrier()

        if multiqubit_entanglement is not None:
            for _ in range(extra_entanglement_layers):
                if adaptive:
                    _rot(rot_index)
                    rot_index += 1
                _apply_multiqubit_entanglement()
                if insert_barriers:
                    qc.barrier()

    if not skip_final_rotation_layer:
        _rot(rot_index)

    qc.input_params = params  # type: ignore[attr-defined]
    qc.num_rot_layers = total_rot_layers  # type: ignore[attr-defined]
    return qc

# --------------------------------------------------------------------------- #
# Convenience wrapper class
# --------------------------------------------------------------------------- #
class RealAmplitudesCZExtended(QuantumCircuit):
    """Convenience wrapper for the depth‑controlled, multi‑entangler RealAmplitudesCZ.

    The class behaves like a normal :class:`~qiskit.circuit.QuantumCircuit` and
    forwards all arguments to :func:`real_amplitudes_cz_extended`.  It also
    exposes the ``input_params`` and ``num_rot_layers`` attributes for
    convenient access during training or analysis.
    """

    def __init__(
        self,
        num_qubits: int,
        reps: int = 1,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
        skip_final_rotation_layer: bool = False,
        insert_barriers: bool = False,
        parameter_prefix: str = "theta",
        name: str = "RealAmplitudesCZExtended",
        multiqubit_entanglement: str | None = None,
        block_size: int = 3,
        extra_entanglement_layers: int = 0,
        adaptive: bool = False,
    ) -> None:
        built = real_amplitudes_cz_extended(
            num_qubits,
            reps,
            entanglement,
            skip_final_rotation_layer,
            insert_barriers,
            parameter_prefix,
            name,
            multiqubit_entanglement,
            block_size,
            extra_entanglement_layers,
            adaptive,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)
        self.input_params = built.input_params  # type: ignore[attr-defined]
        self.num_rot_layers = built.num_rot_layers  # type: ignore[attr-defined]
