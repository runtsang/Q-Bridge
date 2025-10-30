import torch
from torch import nn
import numpy as np

class ConvGen194(nn.Module):
    """
    Depthwise separable convolution with learnable threshold and optional quantum support.
    """

    def __init__(self,
                 kernel_sizes: list[int] = [1, 2, 3],
                 stride: int = 1,
                 padding: int = 0,
                 threshold: float = 0.0,
                 use_quantum: bool = False,
                 quantum_backend=None,
                 shots: int = 100):
        super().__init__()
        self.kernel_sizes = kernel_sizes
        self.stride = stride
        self.padding = padding
        self.threshold = nn.Parameter(torch.tensor(threshold, dtype=torch.float32))
        self.use_quantum = use_quantum
        self.quantum_backend = quantum_backend
        self.shots = shots

        # Build depthwise separable convs for each kernel size
        self.depthwise_convs = nn.ModuleList()
        self.pointwise_convs = nn.ModuleList()
        for k in kernel_sizes:
            depthwise = nn.Conv2d(1, 1, kernel_size=k, stride=stride, padding=padding, groups=1, bias=False)
            pointwise = nn.Conv2d(1, 1, kernel_size=1, bias=True)
            self.depthwise_convs.append(depthwise)
            self.pointwise_convs.append(pointwise)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that combines outputs from multiple kernel sizes and an optional quantum circuit.
        """
        if x.dim() == 2:
            x = x.unsqueeze(0).unsqueeze(0)  # (H, W) -> (1, 1, H, W)

        outputs = []
        for depthwise, pointwise in zip(self.depthwise_convs, self.pointwise_convs):
            out = depthwise(x)
            out = pointwise(out)
            out = torch.sigmoid(out - self.threshold)
            outputs.append(out.mean(dim=(2, 3)))  # mean over spatial dims

        # Average across kernel sizes
        out = torch.stack(outputs, dim=0).mean(dim=0)  # shape: (batch, channels)

        # Optional quantum contribution
        if self.use_quantum and self.quantum_backend is not None:
            # Flatten the input to a 1D array for the quantum circuit
            data_np = x.squeeze().detach().cpu().numpy()
            n_qubits = data_np.size
            # Build a simple variational circuit
            from qiskit import QuantumCircuit, Aer, execute
            qc = QuantumCircuit(n_qubits)
            # Parameterized RX gates
            for i in range(n_qubits):
                angle = np.pi if data_np[i] > self.threshold.item() else 0.0
                qc.rx(angle, i)
            # Entanglement layer
            for i in range(n_qubits - 1):
                qc.cx(i, i + 1)
            qc.measure_all()
            job = execute(qc, self.quantum_backend, shots=self.shots)
            result = job.result()
            counts = result.get_counts()
            prob = 0.0
            for bitstring, count in counts.items():
                ones = bitstring.count('1')
                prob += ones * count
            prob /= (self.shots * n_qubits)
            out += torch.tensor(prob, dtype=torch.float32)

        return out.squeeze()

def Conv(*args, **kwargs):
    """
    Convenience function that returns a ConvGen194 instance.
    """
    return ConvGen194(*args, **kwargs)
