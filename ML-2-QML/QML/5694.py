import numpy as np
import torch
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, Pauli

def quantum_latent_encoder(
    latent: torch.Tensor,
    n_qubits: int,
    n_layers: int,
    ansatz_params: np.ndarray,
) -> torch.Tensor:
    """
    Encodes a classical latent vector into a quantum state using
    RY rotations followed by a RealAmplitudes ansatz.
    Returns the Z‑expectation values of each qubit.
    """
    batch = latent.shape[0]
    results = []
    for i in range(batch):
        vec = latent[i].cpu().numpy()
        qc = QuantumCircuit(n_qubits)
        # Encode classical vector
        for q in range(min(vec.shape[0], n_qubits)):
            theta = vec[q]
            qc.ry(theta, q)
        # RealAmplitudes ansatz
        for rep in range(n_layers):
            for q in range(n_qubits):
                for gate_idx, gate in enumerate(['rx', 'ry', 'rz']):
                    param = ansatz_params[rep, q, gate_idx]
                    if gate == 'rx':
                        qc.rx(param, q)
                    elif gate == 'ry':
                        qc.ry(param, q)
                    else:
                        qc.rz(param, q)
        # Simulate
        state = Statevector.from_int(0, dims=(2**n_qubits,))
        state = state.evolve(qc)
        # Expectation of Z on each qubit
        z_exp = []
        for q in range(n_qubits):
            pauli_z = Pauli('Z')
            exp_val = state.expectation_value(pauli_z, qubit=q)
            z_exp.append(exp_val)
        results.append(z_exp)
    return torch.tensor(np.array(results, dtype=np.float32))

def sample_ansatz_params(n_layers: int, n_qubits: int) -> np.ndarray:
    """
    Generate random initial parameters for the RealAmplitudes ansatz.
    Returns an array of shape (n_layers, n_qubits, 3).
    """
    return np.random.randn(n_layers, n_qubits, 3)

# Example usage
if __name__ == "__main__":
    latent = torch.randn(5, 4)  # batch of 5 latent vectors of dimension 4
    n_qubits = 3
    n_layers = 2
    params = sample_ansatz_params(n_layers, n_qubits)
    out = quantum_latent_encoder(latent, n_qubits, n_layers, params)
    print(out.shape)  # (5, 3)
