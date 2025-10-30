import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.providers.aer import AerSimulator

class SamplerQNN(nn.Module):
    """
    Quantum sampler that encodes input features into a parameterised
    circuit and returns a probability distribution over two classes.
    The circuit is built with an encoding layer followed by a
    depth‑controlled variational ansatz, mirroring the structure
    of the classical SamplerQNN but with a quantum backend.
    """
    def __init__(self, input_dim=2, num_qubits=None, depth=2, shots=1024, backend=None):
        super().__init__()
        self.input_dim = input_dim
        self.num_qubits = num_qubits or input_dim
        self.depth = depth
        self.shots = shots
        self.backend = backend or AerSimulator(method="statevector")

        # Parameter vectors
        self.encoding_params = ParameterVector("x", self.input_dim)
        self.weight_params = ParameterVector("theta", self.num_qubits * self.depth)

        # Build the circuit and observables
        self.circuit, self.observables = self._build_circuit()

        # Trainable variational parameters as torch parameters
        init_weights = np.random.randn(self.num_qubits * self.depth)
        self.weight = nn.Parameter(torch.tensor(init_weights, dtype=torch.float32))

    def _build_circuit(self):
        """Create a layered ansatz with encoding and variational layers."""
        circuit = QuantumCircuit(self.num_qubits)
        # Encoding: RX on each qubit
        for param, qubit in zip(self.encoding_params, range(self.num_qubits)):
            circuit.rx(param, qubit)

        # Variational layers
        idx = 0
        for _ in range(self.depth):
            for qubit in range(self.num_qubits):
                circuit.ry(self.weight_params[idx], qubit)
                idx += 1
            for qubit in range(self.num_qubits - 1):
                circuit.cz(qubit, qubit + 1)

        # Observables: Z on each qubit
        observables = [SparsePauliOp("I" * i + "Z" + "I" * (self.num_qubits - i - 1))
                       for i in range(self.num_qubits)]
        return circuit, observables

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Compute the expectation values of the first two Z observables
        for each input sample and return a soft‑max probability
        distribution over two classes.
        """
        inputs_np = inputs.detach().cpu().numpy()
        batch_size = inputs_np.shape[0]
        logits = np.zeros((batch_size, 2), dtype=np.float32)

        # Bind the variational parameters once
        weight_vals = self.weight.detach().cpu().numpy()
        weight_bind = {p: val for p, val in zip(self.weight_params, weight_vals)}

        for i in range(batch_size):
            bind_dict = {p: val for p, val in zip(self.encoding_params, inputs_np[i])}
            bind_dict.update(weight_bind)
            bound_circ = self.circuit.bind_parameters(bind_dict)
            state = Statevector.from_instruction(bound_circ)
            for j in range(2):
                exp_val = state.expectation_value(self.observables[j]).real
                logits[i, j] = exp_val

        probs = F.softmax(torch.tensor(logits, device=inputs.device), dim=-1)
        return probs

__all__ = ["SamplerQNN"]
