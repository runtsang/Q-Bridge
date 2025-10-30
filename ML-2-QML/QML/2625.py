import numpy as np
import qiskit
from qiskit.circuit.random import random_circuit
import torchquantum as tq
from torchquantum.functional import func_name_dict
from typing import Sequence

class ConvQuantumKernel(tq.QuantumModule):
    """
    Hybrid quantum convolution (quanvolution) + quantum kernel.

    Implements a 2‑D quanvolution filter built from a parameterised circuit
    and a fixed TorchQuantum kernel ansatz.  It can be used as a drop‑in
    replacement for the classical Conv while exposing a kernel_matrix method
    for similarity computations.
    """

    def __init__(self, kernel_size: int = 2, shots: int = 200, threshold: float = 127.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.shots = shots
        self.threshold = threshold
        self.n_qubits = kernel_size ** 2

        # Quanvolution circuit
        self._circuit = qiskit.QuantumCircuit(self.n_qubits)
        self.theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            self._circuit.rx(self.theta[i], i)
        self._circuit.barrier()
        self._circuit += random_circuit(self.n_qubits, 2)
        self._circuit.measure_all()
        self.backend = qiskit.Aer.get_backend("qasm_simulator")

        # Quantum kernel ansatz (Ry on each qubit)
        self.q_device = tq.QuantumDevice(n_wires=self.n_qubits)
        self.kernel_ansatz = tq.QuantumModule()
        # Define a simple Ry ansatz that will be applied in forward
        self.kernel_ansatz.apply = lambda qd, x, y: None  # placeholder; actual logic in forward

    def forward(self, patch: torch.Tensor) -> torch.Tensor:
        """
        Apply the quanvolution filter to a patch.
        """
        if patch.ndim == 2:
            patch = patch.reshape(1, self.n_qubits)
        else:
            patch = patch.reshape(-1, self.n_qubits)
        param_binds = []
        for dat in patch:
            bind = {}
            for i, val in enumerate(dat):
                bind[self.theta[i]] = np.pi if val > self.threshold else 0
            param_binds.append(bind)

        job = qiskit.execute(
            self._circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=param_binds
        )
        result = job.result().get_counts(self._circuit)
        counts = 0
        for key, val in result.items():
            ones = sum(int(bit) for bit in key)
            counts += ones * val
        return torch.tensor(counts / (self.shots * self.n_qubits))

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        """
        Compute a quantum kernel Gram matrix using a simple Ry ansatz.
        """
        a = [x.reshape(-1) for x in a]
        b = [y.reshape(-1) for y in b]
        a = torch.stack(a)
        b = torch.stack(b)

        # Encode data into the device
        self.q_device.reset_states(a.shape[0] + b.shape[0])

        # Forward pass for a
        for i, x in enumerate(a):
            self.q_device.apply(tq.ops.RY, wires=i, params=x)

        # Forward pass for b with negative params
        for i, y in enumerate(b):
            self.q_device.apply(tq.ops.RY, wires=i, params=-y)

        # Compute overlap (inner product of state vectors)
        states = self.q_device.states.view(-1)
        kernel = torch.abs(states[0: a.shape[0]] @ states[a.shape[0]:].conj().T)
        return kernel.detach().numpy()

__all__ = ["ConvQuantumKernel"]
