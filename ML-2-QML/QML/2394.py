import torch
import torch.nn as nn
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit.providers.aer import AerSimulator

class HybridSamplerQNN(nn.Module):
    """Quantum-only component that builds a parameterized circuit and returns
    the probability distribution over basis states."""
    def __init__(self, num_wires: int = 2):
        super().__init__()
        self.num_wires = num_wires
        self.input_params = ParameterVector("input", self.num_wires)
        self.weight_params = ParameterVector("weight", 4 * self.num_wires)
        self.circuit = self._build_circuit()
        self.qsim = AerSimulator()

    def _build_circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.num_wires)
        # Encode inputs
        for i in range(self.num_wires):
            qc.ry(self.input_params[i], i)
        # Entanglement layer
        for i in range(self.num_wires - 1):
            qc.cx(i, i + 1)
        # Parameterized rotations
        for i in range(self.num_wires):
            qc.ry(self.weight_params[i], i)
        # Additional entanglement
        for i in range(self.num_wires - 1):
            qc.cx(i, i + 1)
        for i in range(self.num_wires):
            qc.ry(self.weight_params[self.num_wires + i], i)
        return qc

    def forward(self, input_vals: torch.Tensor, weight_vals: torch.Tensor) -> torch.Tensor:
        """Return probability vector for each sample in the batch.
        input_vals: shape (batch, num_wires)
        weight_vals: shape (batch, 4*num_wires)
        """
        batch_size = input_vals.shape[0]
        probs = torch.empty(batch_size, 2 ** self.num_wires, device=input_vals.device)
        for i in range(batch_size):
            # Bind parameters
            bound_circuit = self.circuit.bind_parameters({
                **{self.input_params[j]: input_vals[i, j].item() for j in range(self.num_wires)},
                **{self.weight_params[j]: weight_vals[i, j].item() for j in range(4 * self.num_wires)},
            })
            # Transpile and run
            transpiled = transpile(bound_circuit, self.qsim)
            result = self.qsim.run(transpiled).result()
            statevector = result.get_statevector()
            probs[i] = torch.tensor(abs(statevector)**2, dtype=torch.float32, device=input_vals.device)
        return probs
