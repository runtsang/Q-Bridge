import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Sequence, List, Tuple, Callable

class HybridSamplerEstimatorQNN(nn.Module):
    """
    Hybrid sampler & estimator network.  It contains a shared hidden layer
    feeding two heads: a soft‑max sampler and a regression estimator.  Optionally
    a quantum LSTM cell can be inserted between the input and the heads for
    quantum‑enhanced gating.  The class also implements a FastEstimator-like
    evaluation interface with optional Gaussian shot noise.
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 4,
        n_qubits: int = 0,
        use_lstm: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        input_dim : int
            Dimension of the input vector.
        hidden_dim : int
            Size of the shared hidden representation.
        n_qubits : int
            Number of qubits for the optional quantum LSTM gating.
        use_lstm : bool
            If True and n_qubits > 0, a quantum LSTM cell is used.
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.use_lstm = use_lstm and n_qubits > 0

        # Optional quantum LSTM gating (fallback to classical linear if not available)
        if self.use_lstm:
            try:
                import torchquantum as tq
                import torchquantum.functional as tqf
            except Exception:
                self.use_lstm = False

        if self.use_lstm:
            class QLayer(tq.QuantumModule):  # type: ignore
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
                    qdev = tq.QuantumDevice(
                        n_wires=self.n_wires,
                        bsz=x.shape[0],
                        device=x.device,
                    )
                    self.encoder(qdev, x)
                    for wire, gate in enumerate(self.params):
                        gate(qdev, wires=wire)
                    for wire in range(self.n_wires):
                        tgt = 0 if wire == self.n_wires - 1 else wire + 1
                        tqf.cnot(qdev, wires=[wire, tgt])
                    return self.measure(qdev)

            self.forget = QLayer(n_qubits)
            self.input = QLayer(n_qubits)
            self.update = QLayer(n_qubits)
            self.output = QLayer(n_qubits)

            self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
            self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
            self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
            self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)
        else:
            # Classical gating: simple linear projection to hidden_dim
            self.linear_forget = nn.Linear(input_dim + hidden_dim, hidden_dim)
            self.linear_input = nn.Linear(input_dim + hidden_dim, hidden_dim)
            self.linear_update = nn.Linear(input_dim + hidden_dim, hidden_dim)
            self.linear_output = nn.Linear(input_dim + hidden_dim, hidden_dim)

        # Shared linear layer before heads
        self.shared = nn.Linear(input_dim, hidden_dim)

        # Sampler head
        self.sampler_head = nn.Sequential(nn.Linear(hidden_dim, 2))

        # Estimator head
        self.estimator_head = nn.Sequential(
            nn.Linear(hidden_dim, 8),
            nn.Tanh(),
            nn.Linear(8, 4),
            nn.Tanh(),
            nn.Linear(4, 1),
        )

    def _lstm_step(
        self,
        x: torch.Tensor,
        hx: torch.Tensor,
        cx: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        combined = torch.cat([x, hx], dim=1)
        f = torch.sigmoid(self.forget(self.linear_forget(combined)))
        i = torch.sigmoid(self.input(self.linear_input(combined)))
        g = torch.tanh(self.update(self.linear_update(combined)))
        o = torch.sigmoid(self.output(self.linear_output(combined)))
        cx = f * cx + i * g
        hx = o * torch.tanh(cx)
        return hx, cx

    def _init_states(
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(0)
        device = inputs.device
        hx = torch.zeros(batch_size, self.hidden_dim, device=device)
        cx = torch.zeros(batch_size, self.hidden_dim, device=device)
        return hx, cx

    def forward(
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> dict:
        """
        Forward pass that returns a dictionary with the sampler probabilities
        and the regression output.
        """
        hx, cx = self._init_states(inputs, states)
        if self.use_lstm:
            for x in inputs.unbind(dim=0):
                hx, cx = self._lstm_step(x.unsqueeze(0), hx, cx)
            gated = hx
        else:
            gated = self.shared(inputs)

        sampler_logits = self.sampler_head(gated)
        sampler_probs = F.softmax(sampler_logits, dim=-1)
        estimator_out = self.estimator_head(gated)
        return {"sampler": sampler_probs, "estimator": estimator_out}

    # ----------------------------------------------------------------------
    # FastEstimator-style evaluation interface
    # ----------------------------------------------------------------------
    def evaluate(
        self,
        parameter_sets: Sequence[Sequence[float]],
        observables: Sequence[Callable[[dict], torch.Tensor | float]] | None = None,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """
        Evaluate the network for a list of input vectors.

        Parameters
        ----------
        parameter_sets : Sequence[Sequence[float]]
            List of input vectors (each a sequence of floats).
        observables : Sequence[Callable[[dict], torch.Tensor | float]] | None
            List of callables that map the output dictionary to a scalar.
            Defaults to sampler mean and estimator value.
        shots : int | None
            If provided, adds Gaussian shot noise with variance 1/shots.
        seed : int | None
            Random seed for shot noise.
        """
        if observables is None:
            observables = [
                lambda out: out["sampler"].mean().item(),
                lambda out: out["estimator"].item(),
            ]

        self.eval()
        results: List[List[float]] = []
        with torch.no_grad():
            for params in parameter_sets:
                inp = torch.as_tensor(params, dtype=torch.float32).unsqueeze(0)
                out = self(inp)
                row = []
                for obs in observables:
                    val = obs(out)
                    if isinstance(val, torch.Tensor):
                        val = val.item()
                    row.append(float(val))
                results.append(row)

        if shots is not None:
            rng = np.random.default_rng(seed)
            noisy = []
            for row in results:
                noisy_row = [
                    rng.normal(mean, max(1e-6, 1 / shots)) for mean in row
                ]
                noisy.append(noisy_row)
            results = noisy

        return results

__all__ = ["HybridSamplerEstimatorQNN"]
