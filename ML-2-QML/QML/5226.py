import torch
import torch.nn as nn
from qiskit import Aer, QuantumCircuit, execute
from qiskit.quantum_info import Statevector, Pauli

class QuantumFCL(nn.Module):
    """Quantum fully‑connected layer returning a scalar expectation."""
    def __init__(self, n_qubits: int) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.backend = Aer.get_backend('statevector_simulator')

    def forward(self, thetas: torch.Tensor) -> torch.Tensor:
        # thetas shape (batch, n_qubits)
        outputs = []
        for theta in thetas:
            circ = QuantumCircuit(self.n_qubits)
            for q in range(self.n_qubits):
                circ.ry(theta[q].item(), q)
            result = execute(circ, self.backend, shots=1).result()
            state = Statevector(result.get_statevector(circ))
            exp = state.expectation_value(Pauli('Z' + 'I' * (self.n_qubits - 1)))
            outputs.append(exp)
        return torch.tensor(outputs, device=thetas.device)

class QLSTMGen108(nn.Module):
    """Hybrid LSTM where gates are evaluated on a small quantum circuit."""
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int = 0) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        # use hidden_dim qubits for the quantum part
        self.n_qubits = hidden_dim

        # linear mapping to gate parameters
        self.linear_forget = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.linear_input = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.linear_update = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.linear_output = nn.Linear(input_dim + hidden_dim, hidden_dim)

        self.backend = Aer.get_backend('statevector_simulator')
        self.fcl = QuantumFCL(n_qubits=hidden_dim)

    def _clip(self, x, bound=5.0):
        return torch.clamp(x, -bound, bound)

    def _init_states(self, inputs, states=None):
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (
            torch.zeros(batch_size, self.hidden_dim, device=device),
            torch.zeros(batch_size, self.hidden_dim, device=device),
        )

    def _quantum_gate(self, params: torch.Tensor) -> torch.Tensor:
        """Evaluate a tiny quantum circuit for each batch element and return
        a vector of Pauli‑Z expectations – one per qubit."""
        outputs = []
        for i in range(params.size(0)):
            circ = QuantumCircuit(self.n_qubits)
            for q in range(self.n_qubits):
                circ.rx(params[i, q].item(), q)
            for q in range(self.n_qubits - 1):
                circ.cx(q, q + 1)
            result = execute(circ, self.backend, shots=1).result()
            state = Statevector(result.get_statevector(circ))
            exp_vec = []
            for q in range(self.n_qubits):
                pauli_str = 'I'*q + 'Z' + 'I'*(self.n_qubits - q - 1)
                exp_vec.append(state.expectation_value(Pauli(pauli_str)))
            outputs.append(exp_vec)
        return torch.tensor(outputs, device=params.device)

    def forward(self, inputs: torch.Tensor, states: tuple[torch.Tensor, torch.Tensor] | None = None):
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)

            f_params = self._clip(self.linear_forget(combined))
            i_params = self._clip(self.linear_input(combined))
            g_params = self._clip(self.linear_update(combined))
            o_params = self._clip(self.linear_output(combined))

            f = torch.sigmoid(self._quantum_gate(f_params))
            i = torch.sigmoid(self._quantum_gate(i_params))
            g = torch.tanh(self._quantum_gate(g_params))
            o = torch.sigmoid(self._quantum_gate(o_params))

            cx = f * cx + i * g
            hx = o * torch.tanh(cx)

            # modulate with quantum FCL expectation
            mod = self.fcl(hx).unsqueeze(-1).expand_as(hx)
            hx = hx * mod

            outputs.append(hx.unsqueeze(0))
        outputs = torch.cat(outputs, dim=0)
        return outputs, (hx, cx)

__all__ = ["QLSTMGen108"]
