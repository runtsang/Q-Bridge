import numpy as np
import torch
import torch.nn as nn
import torch.quantum as tq
from torchquantum.functional import func_name_dict, cnot
from qiskit import QuantumCircuit, Aer, execute
from qiskit.primitives import Sampler
from qiskit_machine_learning.neural_networks import SamplerQNN as QiskitSamplerQNN

# ------------------------------------------------------------------
# Quantum convolutional layer (analogous to a 2×2 quanvolution filter)
# ------------------------------------------------------------------
class QuantumConvLayer(tq.QuantumModule):
    """
    A small parameter‑free quantum circuit that transforms a 2‑qubit
    input patch into expectation values.  It mimics the behaviour of
    the classical ConvFilter but operates on qubits.
    """
    def __init__(self, n_wires: int = 4) -> None:
        super().__init__()
        self.n_wires = n_wires
        # Encode input via Ry gates
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "ry", "wires": [i]} for i in range(n_wires)]
        )
        # Fixed entanglement to introduce non‑linearity
        self.entangler = tq.GeneralEntangler(
            [{"func": "cnot", "wires": [i, (i+1) % n_wires]} for i in range(n_wires)]
        )
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape (batch, n_wires).  Values are treated as rotation angles.
        Returns
        -------
        torch.Tensor
            Expectation values of Pauli‑Z on each qubit, shape (batch, n_wires).
        """
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=x.shape[0], device=x.device)
        self.encoder(qdev, x)
        self.entangler(qdev)
        return self.measure(qdev)

# ------------------------------------------------------------------
# Quantum LSTM cell (using small quantum modules for gates)
# ------------------------------------------------------------------
class QLSTM(nn.Module):
    """
    Quantum LSTM cell where each gate is a tiny quantum circuit.
    This mirrors the classical LSTM but introduces quantum non‑linearities.
    """
    class QGate(tq.QuantumModule):
        def __init__(self, n_wires: int) -> None:
            super().__init__()
            self.n_wires = n_wires
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
            )
            self.params = nn.ModuleList([tq.RX(has_params=True, trainable=True) for _ in range(n_wires)])
            self.entangle = tq.GeneralEntangler(
                [{"func": "cnot", "wires": [i, (i+1)%n_wires]} for i in range(n_wires)]
            )
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=x.shape[0], device=x.device)
            self.encoder(qdev, x)
            for w, gate in enumerate(self.params):
                gate(qdev, wires=w)
            self.entangle(qdev)
            return self.measure(qdev)

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        self.forget = self.QGate(n_qubits)
        self.input = self.QGate(n_qubits)
        self.update = self.QGate(n_qubits)
        self.output = self.QGate(n_qubits)

        self.lin_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.lin_input = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.lin_update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.lin_output = nn.Linear(input_dim + hidden_dim, n_qubits)

    def forward(self,
                inputs: torch.Tensor,
                states: tuple[torch.Tensor, torch.Tensor] | None = None
                ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget(self.lin_forget(combined)))
            i = torch.sigmoid(self.input(self.lin_input(combined)))
            g = torch.tanh(self.update(self.lin_update(combined)))
            o = torch.sigmoid(self.output(self.lin_output(combined)))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        stacked = torch.cat(outputs, dim=0)
        return stacked, (hx, cx)

    def _init_states(self,
                     inputs: torch.Tensor,
                     states: tuple[torch.Tensor, torch.Tensor] | None
                     ) -> tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (torch.zeros(batch_size, self.hidden_dim, device=device),
                torch.zeros(batch_size, self.hidden_dim, device=device))

# ------------------------------------------------------------------
# Quantum kernel (fixed TorchQuantum ansatz)
# ------------------------------------------------------------------
class KernalAnsatz(tq.QuantumModule):
    """
    Encodes two classical vectors into a quantum state and
    computes the overlap, acting as a kernel.
    """
    def __init__(self, func_list: list[dict]) -> None:
        super().__init__()
        self.func_list = func_list

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        q_device.reset_states(x.shape[0])
        for info in self.func_list:
            params = x[:, info["input_idx"]] if func_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)
        for info in reversed(self.func_list):
            params = -y[:, info["input_idx"]] if func_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)

class Kernel(tq.QuantumModule):
    """
    Wrapper that evaluates the quantum kernel on two vectors.
    """
    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.ansatz = KernalAnsatz(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.ansatz(self.q_device, x, y)
        return torch.abs(self.q_device.states.view(-1)[0])

def kernel_matrix(a: list[torch.Tensor], b: list[torch.Tensor]) -> np.ndarray:
    kernel = Kernel()
    return np.array([[kernel(x, y).item() for y in b] for x in a])

# ------------------------------------------------------------------
# Hybrid quantum sampler
# ------------------------------------------------------------------
class HybridSamplerQNN(tq.QuantumModule):
    """
    Quantum‑centric hybrid sampler that processes 4‑dimensional
    sequences through a quantum convolution, a quantum LSTM,
    a quantum kernel, and finally samples from a 2‑qubit variational circuit.
    The public API mirrors the classical HybridSamplerQNN for head‑to‑head tests.
    """
    def __init__(self,
                 input_dim: int = 4,
                 hidden_dim: int = 8,
                 n_qubits: int = 4,
                 conv_kernel_size: int = 2,
                 threshold: float = 0.0,
                 gamma: float = 1.0,
                 backend=None,
                 shots: int = 100) -> None:
        super().__init__()
        if input_dim!= conv_kernel_size ** 2:
            raise ValueError("input_dim must equal conv_kernel_size**2 for the quantum conv.")
        self.quantum_conv = QuantumConvLayer(n_wires=conv_kernel_size**2)
        self.quantum_lstm = QLSTM(input_dim, hidden_dim, n_qubits)
        self.kernel = Kernel()
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.shots = shots
        self.threshold = threshold
        self.gamma = gamma

        # 2‑qubit variational sampler circuit (parameters will be bound later)
        self.sampler_circuit = QuantumCircuit(2)
        self.sampler_circuit.ry(0, 0)
        self.sampler_circuit.ry(0, 1)
        self.sampler_circuit.cz(0, 1)  # entangling

        self.sampler = Sampler(self.backend)

    def _sample_from_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        Run the 2‑qubit sampler with the first two hidden‑state components as Ry angles.
        Returns a probability vector of shape (2,) on the CPU.
        """
        angles = hidden.cpu().numpy()
        params = {self.sampler_circuit.params[0]: angles[0],
                  self.sampler_circuit.params[1]: angles[1]}
        job = self.sampler.run(self.sampler_circuit, parameter_binds=[params], shots=self.shots)
        result = job.get_counts()
        probs = np.zeros(2)
        for outcome, count in result.items():
            idx = int(outcome[::-1], 2)  # bitstring reversed because Qiskit uses little‑endian
            probs[idx] += count
        probs /= self.shots
        return torch.tensor(probs, device=hidden.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape (batch, seq_len, input_dim).  input_dim must be 4.
        Returns
        -------
        torch.Tensor
            Probabilities from the 2‑qubit sampler, shape (batch, 2).
        """
        batch, seq_len, _ = x.shape

        # Quantum convolution: produce a scalar per time step per batch
        conv_out = torch.zeros(batch, seq_len, device=x.device)
        for b in range(batch):
            for t in range(seq_len):
                patch = x[b, t].view(2, 2).cpu().numpy().flatten()
                patch_tensor = torch.tensor(patch, dtype=torch.float32, device=x.device)
                conv_out[b, t] = self.quantum_conv(patch_tensor).mean()

        # Quantum LSTM over the sequence of conv outputs
        lstm_out, _ = self.quantum_lstm(conv_out.unsqueeze(-1))  # input_dim=1
        last_hidden = lstm_out[:, -1, :]  # shape (batch, hidden_dim)

        # Quantum kernel similarity to a zero reference vector
        zero_ref = torch.zeros(batch, x.shape[-1], device=x.device)
        kernel_sim = self.kernel(last_hidden, zero_ref).unsqueeze(-1)  # (batch,1)

        # Sample probabilities; weight by kernel similarity
        probs = torch.stack([self._sample_from_hidden(last_hidden[b]) for b in range(batch)])
        probs = probs * kernel_sim.squeeze(-1).unsqueeze(-1) + 1e-8  # avoid zero
        probs = probs / probs.sum(dim=1, keepdim=True)
        return probs

__all__ = ["HybridSamplerQNN"]
