import numpy as np
import torch
import torch.nn as nn
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import Parameter
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import Estimator as QiskitEstimator

# --------------------------------------------------------------------
# Quantum convolution filter (parameterised RX per pixel)
# --------------------------------------------------------------------
class QuantumConvFilter:
    def __init__(self, kernel_size: int = 2, shots: int = 1024):
        self.kernel_size = kernel_size
        self.n_qubits = kernel_size ** 2
        self.shots = shots
        self.backend = Aer.get_backend("qasm_simulator")
        # create a template circuit with a parameter for each qubit
        self.base_circuit = QuantumCircuit(self.n_qubits)
        self.params = [Parameter(f"theta{i}") for i in range(self.n_qubits)]
        for i, p in enumerate(self.params):
            self.base_circuit.rx(p, i)
        self.base_circuit.measure_all()

    def run(self, data: np.ndarray) -> np.ndarray:
        """Run the filter on a batch of kernel patches.

        Args:
            data: numpy array of shape (batch, kernel, kernel)
        Returns:
            expectation values of PauliZ for each qubit, shape (batch, n_qubits)
        """
        batch = data.shape[0]
        out = np.zeros((batch, self.n_qubits))
        for i in range(batch):
            bind = {p: np.pi * val for p, val in zip(self.params, data[i].flatten())}
            job = execute(self.base_circuit, self.backend, shots=self.shots,
                          parameter_binds=[bind])
            result = job.result()
            counts = result.get_counts()
            probs = []
            for key in sorted(counts):  # keys are bitstrings
                ones = key.count('1')
                probs.append((ones / self.n_qubits) * counts[key] / self.shots)
            out[i] = np.array(probs)
        return out

# --------------------------------------------------------------------
# Classical transformer backbone (identical to the ML variant)
# --------------------------------------------------------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.q_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x, mask=None):
        batch, seq, _ = x.size()
        q = self.q_linear(x).view(batch, seq, self.num_heads, -1).transpose(1, 2)
        k = self.k_linear(x).view(batch, seq, self.num_heads, -1).transpose(1, 2)
        v = self.v_linear(x).view(batch, seq, self.num_heads, -1).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(batch, seq, self.embed_dim)
        return self.out_proj(out)

class FeedForward(nn.Module):
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x):
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.ffn = FeedForward(embed_dim, ffn_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_out = self.attn(self.norm1(x))
        x = x + self.dropout(attn_out)
        ffn_out = self.ffn(self.norm2(x))
        return x + self.dropout(ffn_out)

class PositionalEncoder(nn.Module):
    def __init__(self, embed_dim: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# --------------------------------------------------------------------
# Fraud‑detection style head (classical)
# --------------------------------------------------------------------
class FraudHead(nn.Module):
    def __init__(self, fraud_layers):
        super().__init__()
        modules = []
        for params in fraud_layers:
            weight = torch.tensor([[params.bs_theta, params.bs_phi],
                                   [params.squeeze_r[0], params.squeeze_r[1]]], dtype=torch.float32)
            bias = torch.tensor(params.phases, dtype=torch.float32)
            weight = weight.clamp(-5.0, 5.0)
            bias = bias.clamp(-5.0, 5.0)
            linear = nn.Linear(2, 2)
            with torch.no_grad():
                linear.weight.copy_(weight)
                linear.bias.copy_(bias)
            activation = nn.Tanh()
            scale = torch.tensor(params.displacement_r, dtype=torch.float32)
            shift = torch.tensor(params.displacement_phi, dtype=torch.float32)

            class Layer(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.linear = linear
                    self.activation = activation
                    self.register_buffer("scale", scale)
                    self.register_buffer("shift", shift)

                def forward(self, inputs):
                    outputs = self.activation(self.linear(inputs))
                    return outputs * self.scale + self.shift

            modules.append(Layer())
        modules.append(nn.Linear(2, 1))
        self.net = nn.Sequential(*modules)

    def forward(self, x):
        return self.net(x)

# --------------------------------------------------------------------
# Quantum estimator head
# --------------------------------------------------------------------
class QuantumEstimator(nn.Module):
    def __init__(self):
        super().__init__()
        # simple 1‑qubit circuit with a parameterised Ry
        params = [Parameter("theta")]
        qc = QuantumCircuit(1)
        qc.ry(params[0], 0)
        qc.measure_all()
        self.circuit = qc
        self.estimator = QiskitEstimator()
        self.qnn = EstimatorQNN(circuit=self.circuit,
                                observables=[("Z", 1)],
                                input_params=[params[0]],
                                estimator=self.estimator)

    def forward(self, x):
        # x: (batch, 1)
        preds = []
        for val in x.detach().cpu().numpy().flatten():
            preds.append(self.qnn.predict({"theta": val})[0])
        return torch.tensor(preds, dtype=torch.float32).unsqueeze(1)

# --------------------------------------------------------------------
# Hybrid model (quantum‑enhanced)
# --------------------------------------------------------------------
class HybridModel(nn.Module):
    """Quantum‑enhanced hybrid architecture mirroring the classical variant."""
    def __init__(self,
                 conv_kernel: int = 2,
                 conv_threshold: float = 0.0,
                 embed_dim: int = 32,
                 num_heads: int = 4,
                 num_blocks: int = 2,
                 ffn_dim: int = 64,
                 fraud_layers: List[FraudLayerParameters] = []):
        super().__init__()
        self.quantum_conv = QuantumConvFilter(kernel_size=conv_kernel)
        self.proj = nn.Linear(1, embed_dim)
        self.transformer = nn.Sequential(
            *[TransformerBlock(embed_dim, num_heads, ffn_dim) for _ in range(num_blocks)]
        )
        self.pos_enc = PositionalEncoder(embed_dim)
        self.fraud_head = FraudHead(fraud_layers)
        self.estimator = QuantumEstimator()

    def forward(self, x):
        # x: (batch, 1, H, W)
        batch, _, h, w = x.shape
        # extract 2x2 patches and run quantum convolution
        patches = torch.unfold(x, kernel_size=(2, 2), stride=2)  # (batch, 1, 4, num_patches)
        patches = patches.permute(0, 2, 3, 1).reshape(-1, 2, 2).cpu().numpy()
        q_feats = self.quantum_conv.run(patches)  # (batch*num_patches, 4)
        q_feats = torch.tensor(q_feats, dtype=torch.float32).reshape(batch, -1, 4)
        q_feats = self.proj(q_feats)  # (batch, seq, embed_dim)
        q_feats = self.pos_enc(q_feats)
        seq_emb = self.transformer(q_feats)
        x = seq_emb.mean(dim=1)  # (batch, embed_dim)
        x = self.fraud_head(x)  # (batch, 1)
        x = self.estimator(x)   # (batch, 1)
        return x

__all__ = ["HybridModel", "FraudLayerParameters"]
