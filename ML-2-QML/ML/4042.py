import torch
import torch.nn as nn
import torch.nn.functional as F
from Conv import Conv
from QuantumClassifierModel import build_classifier_circuit

class HybridQLSTM(nn.Module):
    """
    Classical hybrid LSTM that optionally uses a quantum‑style classifier head
    and a convolutional pre‑processor.  It mirrors the API of the original
    :class:`QLSTM` but pulls in the Conv filter and the
    :func:`build_classifier_circuit` helper from the reference pairs.
    """
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int = 0,
                 conv_kernel: int = 2, conv_threshold: float = 0.0,
                 classifier_depth: int = 2) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.n_qubits = n_qubits

        # Convolutional pre‑processor (classical)
        self.conv = Conv()
        self.conv.kernel_size = conv_kernel
        self.conv.threshold = conv_threshold

        # Classical LSTM core
        self.lstm = nn.LSTM(input_dim, hidden_dim)

        # Classical classifier head mirroring the quantum helper
        self.classifier, _, _, _ = build_classifier_circuit(
            num_features=hidden_dim,
            depth=classifier_depth
        )
        # Convert the PyTorch Sequential into a linear layer for logits
        self.classifier_linear = nn.Linear(hidden_dim, 2)

    def forward(self, inputs: torch.Tensor,
                states: tuple[torch.Tensor, torch.Tensor] | None = None
                ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Parameters
        ----------
        inputs : torch.Tensor
            Sequence of shape ``(seq_len, batch, input_dim)``.
        states : optional
            Tuple of ``h_0`` and ``c_0`` for the LSTM.

        Returns
        -------
        logits : torch.Tensor
            Class logits for each time step.
        states : tuple
            Updated LSTM hidden and cell states.
        """
        # Optional convolutional preprocessing – applied only if the
        # input dimension permits a 2×2 reshaping.
        if self.conv is not None and self.input_dim >= 4:
            conv_outs = []
            for t in inputs.unbind(dim=0):
                # reshape embedding to 2×2 matrix and run the filter
                vec = t.detach().cpu().numpy().reshape(2, 2)
                conv_val = self.conv.run(vec)
                conv_outs.append(torch.tensor(conv_val, dtype=t.dtype, device=t.device))
            conv_tensor = torch.stack(conv_outs, dim=0).unsqueeze(-1)
            inputs = torch.cat([inputs, conv_tensor], dim=2)

        lstm_out, (h_n, c_n) = self.lstm(inputs, states)
        logits = self.classifier_linear(lstm_out)
        return logits, (h_n, c_n)

__all__ = ["HybridQLSTM"]
