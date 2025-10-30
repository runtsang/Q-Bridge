import numpy as np
import torch
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.providers.aer import AerSimulator
from qiskit import transpile, assemble

class QuantumEncoderCircuit:
    """Variational circuit that encodes a latent vector into a quantum state
    and returns the expectation value of Pauli‑Z on the first qubit.
    """
    def __init__(self, num_qubits: int, reps: int = 3, seed: int = 42):
        self.num_qubits = num_qubits
        self.reps = reps
        self.seed = seed
        self.backend = AerSimulator()
        self.circuit = self._build_circuit()

    def _build_circuit(self) -> QuantumCircuit:
        qr = QuantumRegister(self.num_qubits)
        qc = QuantumCircuit(qr)
        # Simple parameterised rotation circuit
        for i in range(self.num_qubits):
            qc.ry(f'theta_{i}', qr[i])
        qc.measure_all()
        return qc

    def expectation(self, params: torch.Tensor | np.ndarray) -> torch.Tensor:
        """Return expectation value of Pauli‑Z on first qubit given parameters.
        Supports batch inputs of shape (batch, num_qubits).
        """
        if isinstance(params, torch.Tensor):
            params_np = params.detach().cpu().numpy()
        else:
            params_np = np.asarray(params)
        if params_np.ndim == 1:
            params_np = params_np.reshape(1, -1)

        exps = []
        for row in params_np:
            param_dict = {f'theta_{i}': float(val) for i, val in enumerate(row)}
            bound_qc = self.circuit.bind_parameters(param_dict)
            compiled = transpile(bound_qc, self.backend)
            qobj = assemble(compiled, shots=1024)
            result = self.backend.run(qobj).result()
            counts = result.get_counts()
            exp = 0.0
            for bitstring, cnt in counts.items():
                # bitstring[-1] corresponds to qubit 0 (first qubit)
                first_bit = int(bitstring[-1])
                exp += (1 if first_bit == 0 else -1) * cnt
            exp /= 1024
            exps.append(exp)
        return torch.tensor(exps, dtype=torch.float32)

    def __repr__(self):
        return f"QuantumEncoderCircuit(num_qubits={self.num_qubits}, reps={self.reps})"
