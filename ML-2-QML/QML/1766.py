import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import Parameter
from qiskit.qasm import QasmQobj
from qiskit.quantum_info import Statevector


class QLSTMGen(nn.Module):
    """Quantum LSTM implemented with Qiskit circuits.  It mirrors the
    interface of the classical hybrid implementation so the two can be
    swapped in experiments.
    """

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        # Parameterised quantum circuits for each gate
        self.forget_circuit = self._build_circuit('f')
        self.input_circuit = self._build_circuit('i')
        self.update_circuit = self._build_circuit('u')
        self.output_circuit = self._build_circuit('o')

        # Linear layers that feed the quantum circuits
        self.forget_linear = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.input_linear = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.update_linear = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.output_linear = nn.Linear(input_dim + hidden_dim, n_qubits)

        # Map quantum outputs to hidden dimension
        self.q_to_hidden = nn.Linear(n_qubits, hidden_dim)

        # Simulator backend
        self.backend = Aer.get_backend('statevector_simulator')

    def _build_circuit(self, name: str) -> QuantumCircuit:
        qc = QuantumCircuit(self.n_qubits)
        # Parameterised RX rotations for each qubit
        for i in range(self.n_qubits):
            theta = Parameter(f'θ_{name}_{i}')
            qc.rx(theta, i)
        # Simple entanglement chain
        for i in range(self.n_qubits - 1):
            qc.cx(i, i + 1)
        return qc

    def _run_circuit(self, qc: QuantumCircuit, params: torch.Tensor) -> torch.Tensor:
        # Bind parameters
        bound_qc = qc.bind_parameters({p: v for p, v in zip(qc.parameters, params.flatten().cpu().numpy())})
        job = execute(bound_qc, self.backend, shots=1024)
        result = job.result()
        statevec = np.array(result.get_statevector(bound_qc))
        # Expectation of Pauli-Z for each qubit
        expectations = []
        for q in range(self.n_qubits):
            prob_one = 0.0
            for idx, amp in enumerate(statevec):
                if (idx >> q) & 1:
                    prob_one += abs(amp) ** 2
            expectations.append(2 * prob_one - 1)  # map to [-1, 1]
        return torch.tensor(expectations, dtype=torch.float32, device=params.device)

    def forward(self, inputs: torch.Tensor, states: tuple = None):
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f_params = self.forget_linear(combined)
            i_params = self.input_linear(combined)
            u_params = self.update_linear(combined)
            o_params = self.output_linear(combined)

            f = torch.sigmoid(self._run_circuit(self.forget_circuit, f_params))
            i = torch.sigmoid(self._run_circuit(self.input_circuit, i_params))
            u = torch.tanh(self._run_circuit(self.update_circuit, u_params))
            o = torch.sigmoid(self._run_circuit(self.output_circuit, o_params))

            f = self.q_to_hidden(f)
            i = self.q_to_hidden(i)
            u = self.q_to_hidden(u)
            o = self.q_to_hidden(o)

            cx = f * cx + i * u
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        return torch.cat(outputs, dim=0), (hx, cx)

    def _init_states(self, inputs, states):
        if states is not None:
            return states
        batch = inputs.size(1)
        device = inputs.device
        return torch.zeros(batch, self.hidden_dim, device=device), torch.zeros(batch, self.hidden_dim, device=device)

__all__ = ["QLSTMGen"]
