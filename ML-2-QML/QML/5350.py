import numpy as np
import torch
import torch.nn as nn
import qiskit
from qiskit import Aer, QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import transpile, assemble
from qiskit.primitives import Sampler

class QuantumSelfAttention:
    """Quantum self‑attention circuit parameterized by rotation and entangle angles."""
    def __init__(self, n_qubits: int, backend=None, shots: int = 1024):
        self.n_qubits = n_qubits
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.shots = shots
        self.rotation_params = np.random.uniform(0, 2*np.pi, size=3*n_qubits)
        self.entangle_params = np.random.uniform(0, np.pi/4, size=n_qubits-1)

    def build(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.n_qubits)
        for i in range(self.n_qubits):
            qc.rx(self.rotation_params[3*i], i)
            qc.ry(self.rotation_params[3*i+1], i)
            qc.rz(self.rotation_params[3*i+2], i)
        for i in range(self.n_qubits-1):
            qc.crx(self.entangle_params[i], i, i+1)
        return qc

class QuantumAutoEncoder:
    """Variational auto‑encoder circuit with a simple ansatz."""
    def __init__(self, n_qubits: int, backend=None, shots: int = 1024):
        self.n_qubits = n_qubits
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.shots = shots

    def build(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.n_qubits)
        for _ in range(2):
            for i in range(self.n_qubits):
                qc.rx(np.random.uniform(0, 2*np.pi), i)
            for i in range(self.n_qubits-1):
                qc.cx(i, i+1)
        return qc

class QuantumExpectationHead:
    """Quantum expectation layer that returns a single logit."""
    def __init__(self, n_qubits: int, backend=None, shots: int = 1024):
        self.n_qubits = n_qubits
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.shots = shots

    def build(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.n_qubits)
        qc.h(range(self.n_qubits))
        qc.barrier()
        qc.ry(np.pi/2, range(self.n_qubits))
        qc.measure_all()
        return qc

    def run(self, circuit: QuantumCircuit) -> float:
        compiled = transpile(circuit, self.backend)
        qobj = assemble(compiled, shots=self.shots)
        job = self.backend.run(qobj)
        result = job.result().get_counts()
        # Expectation of Z on all qubits (simplified)
        counts = np.array(list(result.values()))
        states = np.array([int(k, 2) for k in result.keys()])
        probs = counts / self.shots
        return np.sum(states * probs)

class UnifiedSelfAttentionAutoEncoderHybrid:
    """
    Quantum implementation that mirrors the classical hybrid module.
    It exposes a forward method that accepts a classical tensor,
    runs the quantum self‑attention and auto‑encoder circuits,
    and returns a binary probability pair.
    """
    def __init__(self, n_qubits: int = 4, backend=None, shots: int = 1024):
        self.n_qubits = n_qubits
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.shots = shots
        self.attn = QuantumSelfAttention(n_qubits, backend=self.backend, shots=shots)
        self.ae = QuantumAutoEncoder(n_qubits, backend=self.backend, shots=shots)
        self.head = QuantumExpectationHead(n_qubits, backend=self.backend, shots=shots)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: classical inputs → quantum self‑attention → auto‑encoder → expectation.
        The inputs are ignored in this simplified example and only the circuit
        parameters influence the output, mimicking a quantum‑classical hybrid.
        """
        # Build circuits
        attn_circ = self.attn.build()
        ae_circ = self.ae.build()
        head_circ = self.head.build()

        # Run circuits sequentially (results not used to feed next circuit in this toy demo)
        _ = self.attn.backend.run(assemble(transpile(attn_circ, self.backend), shots=self.shots)).result()
        _ = self.ae.backend.run(assemble(transpile(ae_circ, self.backend), shots=self.shots)).result()

        # Expectation head gives a single logit
        logit = self.head.run(head_circ)
        probs = torch.sigmoid(torch.tensor(logit, dtype=torch.float32))
        return torch.cat((probs, 1 - probs), dim=-1)

__all__ = ["UnifiedSelfAttentionAutoEncoderHybrid"]
