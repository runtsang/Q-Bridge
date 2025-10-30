import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
from torchquantum.functional import func_name_dict

# ------------------------------------------------------------------
#  Quantum convolutional filter – replaces the classical ConvFilter
# ------------------------------------------------------------------
class QuantumConvFilter(tq.QuantumModule):
    """
    Implements a small quanvolution filter.  The circuit consists of a
    parameterised RX rotation for each qubit followed by a randomly
    generated two‑layer circuit.  The output is the average probability
    of measuring |1⟩ across all qubits.
    """
    def __init__(self,
                 kernel_size: int = 2,
                 threshold: float = 0.5,
                 shots: int = 100):
        super().__init__()
        self.n_qubits = kernel_size ** 2
        self.threshold = threshold
        self.shots = shots
        self.circuit = tq.QuantumCircuit(self.n_qubits)
        self.theta = [tq.Parameter(f"theta{i}") for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            self.circuit.rx(self.theta[i], i)
        self.circuit.barrier()
        self.circuit += tq.random_circuit(self.n_qubits, 2)
        self.circuit.measure_all()
        self.backend = tq.AerBackend()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape (batch, seq_len, embed_dim).  embed_dim must be divisible
            by kernel_size**2.
        Returns
        -------
        torch.Tensor
            Shape (batch, seq_len, 1) – scalar per token after quantum
            convolution.
        """
        bs, seq_len, embed_dim = x.shape
        k = int(self.n_qubits ** 0.5)
        flat = x.view(bs * seq_len, k, k)
        probs = []

        for idx in range(bs * seq_len):
            instance = flat[idx]
            param_binds = {}
            for i, val in enumerate(instance.flatten()):
                param_binds[self.theta[i]] = torch.pi if val > self.threshold else 0.0
            job = tq.execute(self.circuit,
                             backend=self.backend,
                             shots=self.shots,
                             parameter_binds=param_binds)
            result = job.get_counts()
            # take the first (and only) key – a bitstring
            bitstring = list(result.keys())[0]
            ones = sum(int(bit) for bit in bitstring)
            probs.append(ones / self.n_qubits)

        probs = torch.tensor(probs, device=x.device)
        return probs.view(bs, seq_len, 1)

# ------------------------------------------------------------------
#  Quantum kernel – directly evaluates a quantum RBF‑style kernel
# ------------------------------------------------------------------
class QuantumKernel(tq.QuantumModule):
    """
    Implements the quantum kernel used in the reference
    implementation.  The ansatz encodes the two inputs with
    ry‑rotations and performs a reverse sweep to compute the overlap.
    """
    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.ansatz = tq.QuantumModule()
        # Build the ansatz
        self.ansatz.add_layer(
            tq.RY, wires=[0], input_idx=[0], num_params=1)
        self.ansatz.add_layer(
            tq.RY, wires=[1], input_idx=[1], num_params=1)
        self.ansatz.add_layer(
            tq.RY, wires=[2], input_idx=[2], num_params=1)
        self.ansatz.add_layer(
            tq.RY, wires=[3], input_idx=[3], num_params=1)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x, y : torch.Tensor
            Shape (batch, 4).
        Returns
        -------
        torch.Tensor
            Shape (batch, 1) – absolute overlap of the two encoded states.
        """
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.ansatz(self.q_device, x, y)
        return torch.abs(self.q_device.states.view(-1)[0]).unsqueeze(0)

# ------------------------------------------------------------------
#  Quantum LSTM cell – gates realised by small quantum circuits
# ------------------------------------------------------------------
class QLayer(tq.QuantumModule):
    """
    Small quantum circuit that implements a single gate of the LSTM.
    """
    def __init__(self, n_wires: int):
        super().__init__()
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "rx", "wires": [0]},
                {"input_idx": [1], "func": "rx", "wires": [1]},
                {"input_idx": [2], "func": "rx", "wires": [2]},
                {"input_idx": [3], "func": "rx", "wires": [3]},
            ]
        )
        self.params = nn.ModuleList(
            [tq.RX(has_params=True, trainable=True) for _ in range(n_wires)]
        )
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=x.shape[0], device=x.device)
        self.encoder(qdev, x)
        for wire, gate in enumerate(self.params):
            gate(qdev, wires=wire)
        for wire in range(self.n_wires):
            if wire == self.n_wires - 1:
                tqf.cnot(qdev, wires=[wire, 0])
            else:
                tqf.cnot(qdev, wires=[wire, wire + 1])
        return self.measure(qdev)

# ------------------------------------------------------------------
#  Quantum‑augmented LSTM – combines classical linear layers with quantum gates
# ------------------------------------------------------------------
class QuantumHybridQLSTM(nn.Module):
    """
    LSTM cell that uses classical linear layers to map the
    concatenated input and hidden state to the number of qubits,
    then feeds the result into a quantum circuit to produce each gate.
    Convolutional preprocessing is performed by QuantumConvFilter.
    """
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 n_qubits: int = 4,
                 conv_kernel: int = 2,
                 kernel_gamma: float = 1.0):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        # Quantum convolution
        self.conv = QuantumConvFilter(kernel_size=conv_kernel,
                                      threshold=0.5,
                                      shots=100)

        # Quantum gates
        self.forget = QLayer(n_qubits)
        self.input = QLayer(n_qubits)
        self.update = QLayer(n_qubits)
        self.output = QLayer(n_qubits)

        # Classical linear mapping to qubit space
        self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)

        # Quantum kernel for regularisation
        self.kernel = QuantumKernel()

    def forward(self,
                inputs: torch.Tensor,
                states: tuple[torch.Tensor, torch.Tensor] | None = None):
        hx, cx = self._init_states(inputs, states)
        outputs = []

        for x in inputs.unbind(dim=0):
            conv_out = self.conv(x.unsqueeze(0)).squeeze(0)
            combined = torch.cat([conv_out, hx], dim=1)

            f = torch.sigmoid(self.forget(self.linear_forget(combined)))
            i = torch.sigmoid(self.input(self.linear_input(combined)))
            g = torch.tanh(self.update(self.linear_update(combined)))
            o = torch.sigmoid(self.output(self.linear_output(combined)))

            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))

        stacked = torch.cat(outputs, dim=0)
        return stacked, (hx, cx)

    def _init_states(self,
                     inputs: torch.Tensor,
                     states: tuple[torch.Tensor, torch.Tensor] | None):
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (torch.zeros(batch_size, self.hidden_dim, device=device),
                torch.zeros(batch_size, self.hidden_dim, device=device))

# ------------------------------------------------------------------
#  Sequence tagging model – uses the quantum‑enhanced LSTM
# ------------------------------------------------------------------
class LSTMTagger(nn.Module):
    """
    Tagger that uses the QuantumHybridQLSTM.  The constructor
    accepts parameters for the quantum part but falls back to a
    classical LSTM if `n_qubits == 0`.
    """
    def __init__(self,
                 embedding_dim: int,
                 hidden_dim: int,
                 vocab_size: int,
                 tagset_size: int,
                 n_qubits: int = 4,
                 conv_kernel: int = 2,
                 kernel_gamma: float = 1.0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits > 0:
            self.lstm = QuantumHybridQLSTM(embedding_dim,
                                           hidden_dim,
                                           n_qubits=n_qubits,
                                           conv_kernel=conv_kernel,
                                           kernel_gamma=kernel_gamma)
        else:
            # fall back to classical HybridQLSTM
            self.lstm = HybridQLSTM(embedding_dim,
                                    hidden_dim,
                                    n_qubits=n_qubits,
                                    conv_kernel=conv_kernel,
                                    kernel_gamma=kernel_gamma)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.embedding(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)

__all__ = ["HybridQLSTM", "LSTMTagger"]
