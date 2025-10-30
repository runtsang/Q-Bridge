"""
Quantum patch filter implemented in Qiskit.

The circuit mirrors the torchquantum implementation used in the hybrid
model, consisting of:
  * 4 RY rotations encoding the 2×2 patch.
  * A fixed random two‑qubit layer.
  * A trainable block (RX, RY, RZ, CRX).
"""

import numpy as np
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import RY, RX, RZ, CRX, CX, CZ, CY
from qiskit.providers.aer import AerSimulator
from typing import Tuple

def build_patch_filter(num_qubits: int = 4, n_random_ops: int = 8, seed: int | None = None) -> QuantumCircuit:
    """
    Build a parameterised circuit that operates on a 4‑qubit register
    representing a 2×2 image patch.

    Parameters
    ----------
    num_qubits : int
        Number of qubits (default 4).
    n_random_ops : int
        Number of random two‑qubit gates in the fixed random layer.
    seed : int | None
        Random seed for reproducibility of the random layer.

    Returns
    -------
    QuantumCircuit
        Parameterised circuit ready for simulation or optimisation.
    """
    qc = QuantumCircuit(num_qubits)

    # Encode the 4 pixel values with RY rotations (parameters will be bound later)
    enc_params = ParameterVector("enc", length=num_qubits)
    for i in range(num_qubits):
        qc.append(RY(enc_params[i]), [i])

    # Fixed random two‑qubit gate layer
    rng = np.random.default_rng(seed)
    for _ in range(n_random_ops):
        q1, q2 = rng.choice(num_qubits, 2, replace=False)
        gate_type = rng.choice(["cx", "cz", "cy"])
        if gate_type == "cx":
            qc.cx(q1, q2)
        elif gate_type == "cz":
            qc.cz(q1, q2)
        else:
            qc.cy(q1, q2)

    # Trainable block – 4 parameters
    train_params = ParameterVector("train", length=4)
    qc.rx(train_params[0], 0)
    qc.ry(train_params[1], 1)
    qc.rz(train_params[2], 3)
    qc.crx(train_params[3], 0, 2)

    return qc

def simulate_patch_filter(circuit: QuantumCircuit,
                          enc_vals: np.ndarray,
                          train_vals: np.ndarray,
                          shots: int = 1024) -> np.ndarray:
    """
    Simulate the patch filter for a batch of patches.

    Parameters
    ----------
    circuit : QuantumCircuit
        Parameterised circuit returned by build_patch_filter.
    enc_vals : np.ndarray
        Array of shape (batch, 4) with pixel values to encode.
    train_vals : np.ndarray
        Array of shape (4,) with trainable parameters.
    shots : int
        Number of shots for the simulation.

    Returns
    -------
    np.ndarray
        Expectation values of Pauli‑Z on all qubits for each patch.
        Shape: (batch, num_qubits)
    """
    backend = AerSimulator()
    results = []
    for enc in enc_vals:
        # Bind parameters for this patch
        param_dict = {circuit.parameters[i]: val for i, val in enumerate(np.concatenate([enc, train_vals]))}
        bound_qc = circuit.bind_parameters(param_dict)

        job = execute(bound_qc, backend, shots=shots, memory=True)
        result = job.result()
        counts = result.get_counts(bound_qc)

        # Convert counts to expectation values of Z
        exp_vals = np.zeros(circuit.num_qubits)
        for key, cnt in counts.items():
            for q in range(circuit.num_qubits):
                bit = int(key[::-1][q])
                exp_vals[q] += cnt * (1 - 2 * bit)  # +1 for 0, -1 for 1
        exp_vals /= shots
        results.append(exp_vals)

    return np.stack(results, axis=0)

__all__ = ["build_patch_filter", "simulate_patch_filter"]
