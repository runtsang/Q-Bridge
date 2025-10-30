from __future__ import annotations

import numpy as np
from typing import Iterable, Sequence
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import RealAmplitudes

class HybridLayer:
    """Quantum hybrid module mirroring HybridLayer in the classical implementation."""
    def __init__(self, mode: str, **kwargs):
        self.mode = mode
        self.backend = Aer.get_backend("qasm_simulator")
        self.shots = kwargs.get("shots", 1024)

        if mode == "fcl":
            n_qubits = kwargs.get("n_qubits", 1)
            theta = ParameterVector("theta", n_qubits)
            qc = QuantumCircuit(n_qubits)
            qc.h(range(n_qubits))
            for i in range(n_qubits):
                qc.ry(theta[i], i)
            qc.measure_all()
            self.circuit = qc
            self.params = theta

        elif mode == "graph":
            arch = kwargs.get("arch", [2, 2, 1])
            n_qubits = arch[-1]
            total_params = sum(arch[:-1]) * n_qubits
            theta = ParameterVector("theta", total_params)
            qc = QuantumCircuit(n_qubits)
            idx = 0
            for i in range(len(arch) - 1):
                out_q = arch[i + 1]
                for q in range(out_q):
                    qc.ry(theta[idx], q)
                    idx += 1
            qc.measure_all()
            self.circuit = qc
            self.params = theta

        elif mode == "autoencoder":
            num_latent = kwargs.get("num_latent", 3)
            num_trash = kwargs.get("num_trash", 2)
            qr = QuantumRegister(num_latent + 2 * num_trash + 1, "q")
            cr = ClassicalRegister(1, "c")
            qc = QuantumCircuit(qr, cr)
            ansatz = RealAmplitudes(num_latent + num_trash, reps=5)
            qc.append(ansatz, range(0, num_latent + num_trash))
            qc.barrier()
            aux = num_latent + 2 * num_trash
            qc.h(aux)
            for i in range(num_trash):
                qc.cswap(aux, num_latent + i, num_latent + num_trash + i)
            qc.h(aux)
            qc.measure(aux, cr[0])
            self.circuit = qc
            self.params = ansatz.parameters

        elif mode == "lstm":
            n_qubits = kwargs.get("n_qubits", 4)
            qr = QuantumRegister(n_qubits, "q")
            cr = ClassicalRegister(1, "c")
            qc = QuantumCircuit(qr, cr)
            for gate_name in ["forget", "input", "update", "output"]:
                theta = ParameterVector(f"{gate_name}_theta", n_qubits)
                qc.h(range(n_qubits))
                for i in range(n_qubits):
                    qc.ry(theta[i], i)
                qc.measure_all()
            self.circuit = qc
            self.params = qc.parameters

        else:
            raise ValueError(f"Unsupported mode {mode}")

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        param_bind = {p: t for p, t in zip(self.params, thetas)}
        job = execute(
            self.circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=[param_bind],
        )
        result = job.result()
        counts = result.get_counts(self.circuit)
        probs = np.array(list(counts.values())) / self.shots
        states = np.array([int(k, 2) for k in counts.keys()])
        expectation = np.sum(states * probs)
        return np.array([expectation])

def FCL(n_qubits: int = 1, shots: int = 1024) -> HybridLayer:
    """Return a quantum fully‑connected layer instance."""
    return HybridLayer(mode="fcl", n_qubits=n_qubits, shots=shots)

def GraphQNN(arch: Sequence[int] = (2, 2, 1), shots: int = 1024) -> HybridLayer:
    """Return a quantum graph‑based neural network instance."""
    return HybridLayer(mode="graph", arch=arch, shots=shots)

def Autoencoder(num_latent: int = 3, num_trash: int = 2, shots: int = 1024) -> HybridLayer:
    """Return a quantum auto‑encoder instance."""
    return HybridLayer(mode="autoencoder", num_latent=num_latent, num_trash=num_trash, shots=shots)

def QLSTM(n_qubits: int = 4, shots: int = 1024) -> HybridLayer:
    """Return a quantum LSTM‑cell instance."""
    return HybridLayer(mode="lstm", n_qubits=n_qubits, shots=shots)

__all__ = ["HybridLayer", "FCL", "GraphQNN", "Autoencoder", "QLSTM"]
