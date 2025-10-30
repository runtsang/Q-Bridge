import numpy as np
import qiskit
from qiskit import execute, assemble, transpile
import torch

# --------------------------------------------------------------------------- #
# Quantum convolution wrapper
# --------------------------------------------------------------------------- #
class QuantumCircuit:
    def __init__(self, n_qubits: int, backend, shots: int) -> None:
        self._circuit = qiskit.QuantumCircuit(n_qubits)
        all_qubits = list(range(n_qubits))
        self.theta = qiskit.circuit.Parameter("theta")
        self._circuit.h(all_qubits)
        self._circuit.barrier()
        self._circuit.ry(self.theta, all_qubits)
        self._circuit.measure_all()
        self.backend = backend
        self.shots = shots

    def run(self, thetas: np.ndarray) -> np.ndarray:
        compiled = transpile(self._circuit, self.backend)
        qobj = assemble(compiled, shots=self.shots,
                         parameter_binds=[{self.theta: theta} for theta in thetas])
        job = self.backend.run(qobj)
        result = job.result().get_counts()

        def expectation(count_dict):
            counts = np.array(list(count_dict.values()))
            states = np.array(list(count_dict.keys())).astype(float)
            probs  = counts / self.shots
            return np.sum(states * probs)

        return np.array([expectation(result)])

# --------------------------------------------------------------------------- #
# Quantum self‑attention block
# --------------------------------------------------------------------------- #
class QuantumSelfAttention:
    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.qr = qiskit.QuantumRegister(n_qubits, "q")
        self.cr = qiskit.ClassicalRegister(n_qubits, "c")

    def _build_circuit(self, rotation_params: np.ndarray, entangle_params: np.ndarray) -> qiskit.QuantumCircuit:
        circuit = qiskit.QuantumCircuit(self.qr, self.cr)
        for i in range(self.n_qubits):
            circuit.rx(rotation_params[3*i], i)
            circuit.ry(rotation_params[3*i+1], i)
            circuit.rz(rotation_params[3*i+2], i)
        for i in range(self.n_qubits-1):
            circuit.crx(entangle_params[i], i, i+1)
        circuit.measure(self.qr, self.cr)
        return circuit

    def run(self, backend, rotation_params, entangle_params, shots: int = 1024):
        circuit = self._build_circuit(rotation_params, entangle_params)
        job = qiskit.execute(circuit, backend, shots=shots)
        return job.result().get_counts(circuit)

# --------------------------------------------------------------------------- #
# Unified hybrid component (quantum side)
# --------------------------------------------------------------------------- #
class ConvHybrid:
    """
    Quantum‑centric counterpart to the classical ConvHybrid.
    Operates a two‑qubit expectation head after a convolution‑style circuit
    and an optional self‑attention block.
    """
    def __init__(
        self,
        kernel_size: int = 2,
        threshold: float = 0.0,
        use_attention: bool = True,
        mode: str = "quantum"  # only quantum mode is supported here
    ) -> None:
        self.mode = mode
        if mode == "quantum":
            self.conv = QuantumCircuit(kernel_size**2,
                                       qiskit.Aer.get_backend("aer_simulator"),
                                       shots=100)
        else:
            raise ValueError("QuantumConvHybrid only supports quantum mode.")
        self.use_attention = use_attention
        if use_attention:
            self.attn = QuantumSelfAttention(n_qubits=kernel_size)

    def forward(self, x: torch.Tensor | np.ndarray) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor or np.ndarray
            2‑D array (H, W) or a batch of such arrays.

        Returns
        -------
        torch.Tensor
            Expectation values from the quantum circuit (optionally processed by attention).
        """
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        if x.ndim == 2:
            x = x.unsqueeze(0)
        outputs = []
        for sample in x:
            data = sample.squeeze().numpy()
            feat = self.conv.run(data.flatten())
            if self.use_attention:
                counts = self.attn.run(qiskit.Aer.get_backend("qasm_simulator"),
                                       np.array([0]), np.array([0]))
                # simple expectation from counts
                total = sum(int(k, 2) * v for k, v in counts.items())
                feat = np.array([total / (len(counts) * self.attn.n_qubits)])
            outputs.append(torch.tensor(feat))
        return torch.stack(outputs)

__all__ = ["ConvHybrid"]
