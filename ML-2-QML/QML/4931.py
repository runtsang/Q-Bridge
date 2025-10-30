import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
from qiskit import QuantumCircuit as QC, assemble, transpile
from qiskit.providers.aer import AerSimulator


class QuantumCircuitWrapper:
    """Parametrised two‑qubit circuit executed on AerSimulator."""
    def __init__(self, n_qubits: int = 2, shots: int = 100):
        self.backend = AerSimulator()
        self.shots = shots
        self.circuit = QC(n_qubits)
        self.theta = QC.circuit.Parameter("theta")
        self.circuit.h(range(n_qubits))
        self.circuit.barrier()
        self.circuit.ry(self.theta, range(n_qubits))
        self.circuit.measure_all()

    def run(self, thetas: np.ndarray) -> np.ndarray:
        compiled = transpile(self.circuit, self.backend)
        qobj = assemble(
            compiled,
            shots=self.shots,
            parameter_binds=[{self.theta: theta} for theta in thetas],
        )
        job = self.backend.run(qobj)
        result = job.result().get_counts()

        def expectation(count_dict):
            counts = np.array(list(count_dict.values()))
            states = np.array([int(k, 2) for k in count_dict.keys()])
            probs = counts / self.shots
            return np.sum(states * probs)

        if isinstance(result, list):
            return np.array([expectation(r) for r in result])
        return np.array([expectation(result)])


class HybridFunction(torch.autograd.Function):
    """Differentiable interface between PyTorch and the quantum circuit."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuitWrapper, shift: float) -> torch.Tensor:
        ctx.circuit = circuit
        ctx.shift = shift
        exp = ctx.circuit.run(inputs.detach().cpu().numpy())
        out = torch.tensor(exp, device=inputs.device, dtype=torch.float32)
        ctx.save_for_backward(inputs, out)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        shift = torch.ones_like(inputs) * ctx.shift
        grads = []
        for val in inputs.detach().cpu().numpy():
            r = ctx.circuit.run([val + shift.item()])
            l = ctx.circuit.run([val - shift.item()])
            grads.append(r - l)
        grads = torch.tensor(grads, device=inputs.device, dtype=torch.float32)
        return grads * grad_output, None, None


class HybridLayer(nn.Module):
    """Hybrid layer that forwards activations through a quantum circuit."""
    def __init__(self, n_qubits: int = 2, shots: int = 100, shift: float = np.pi / 2):
        super().__init__()
        self.circuit = QuantumCircuitWrapper(n_qubits, shots)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return HybridFunction.apply(inputs, self.circuit, self.shift)


class QuantumRegression(tq.QuantumModule):
    """Quantum regression block from reference 2."""
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(n_wires)))
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)

        def forward(self, qdev: tq.QuantumDevice):
            self.random_layer(qdev)
            for w in range(self.n_wires):
                self.rx(qdev, wires=w)
                self.ry(qdev, wires=w)

    def __init__(self, n_wires: int = 4):
        super().__init__()
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{n_wires}xRy"])
        self.q_layer = self.QLayer(n_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(n_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        self.q_layer(qdev)
        feat = self.measure(qdev)
        return self.head(feat).squeeze(-1)


class QuantumNAT(tq.QuantumModule):
    """Quantum‑NAT inspired fully‑connected layer from reference 4."""
    class QLayer(tq.QuantumModule):
        def __init__(self):
            super().__init__()
            self.n_wires = 4
            self.random_layer = tq.RandomLayer(n_ops=50, wires=list(range(self.n_wires)))
            self.rx0 = tq.RX(has_params=True, trainable=True)
            self.ry0 = tq.RY(has_params=True, trainable=True)
            self.rz0 = tq.RZ(has_params=True, trainable=True)
            self.crx0 = tq.CRX(has_params=True, trainable=True)

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice):
            self.random_layer(qdev)
            self.rx0(qdev, wires=0)
            self.ry0(qdev, wires=1)
            self.rz0(qdev, wires=3)
            self.crx0(qdev, wires=[0, 2])
            tqf.hadamard(qdev, wires=3, static=self.static_mode, parent_graph=self.graph)
            tqf.sx(qdev, wires=2, static=self.static_mode, parent_graph=self.graph)
            tqf.cnot(qdev, wires=[3, 0], static=self.static_mode, parent_graph=self.graph)

    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.q_layer = self.QLayer()
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(self.n_wires)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True)
        pooled = F.avg_pool2d(x, 6).view(bsz, 16)
        self.encoder(qdev, pooled)
        self.q_layer(qdev)
        out = self.measure(qdev)
        return self.norm(out)


class SamplerQNNCombined(nn.Module):
    """Quantum‑centric sampler network that fuses classical sampling,
    quantum regression, a hybrid expectation head, and a Quantum‑NAT layer."""
    def __init__(self, n_qubits: int = 2, n_wires: int = 4):
        super().__init__()
        # Classical sampler
        self.sampler = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 2),
        )
        # Quantum regression block
        self.q_regressor = QuantumRegression(n_wires)
        # Hybrid expectation layer
        self.hybrid = HybridLayer(n_qubits)
        # Quantum‑NAT fully‑connected module
        self.q_nat = QuantumNAT()

    def forward(self, inputs: torch.Tensor) -> dict:
        """
        Parameters
        ----------
        inputs : torch.Tensor
            Tensor of shape (batch, 2) representing two classical features.

        Returns
        -------
        dict
            Dictionary containing:
                - ``sampler``: classical soft‑max probabilities,
                - ``regression``: scalar regression output from quantum block,
                - ``hybrid``: expectation value from the hybrid circuit,
                - ``nat``: 4‑dimensional output from Quantum‑NAT.
        """
        # Classical sampler
        sampler_out = F.softmax(self.sampler(inputs), dim=-1)

        # Quantum regression: treat sampler output as state batch
        regression_out = self.q_regressor(sampler_out)

        # Hybrid expectation: feed regression scalar into quantum circuit
        hybrid_out = self.hybrid(regression_out.unsqueeze(-1))

        # Quantum‑NAT: feed hybrid output as 1‑channel image (batch, 1, 1, 1)
        nat_in = hybrid_out.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        nat_out = self.q_nat(nat_in)

        return {
            "sampler": sampler_out,
            "regression": regression_out,
            "hybrid": hybrid_out,
            "nat": nat_out
        }


__all__ = ["SamplerQNNCombined"]
