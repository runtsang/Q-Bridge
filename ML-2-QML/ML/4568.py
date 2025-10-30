"""Self‑attention module that combines classical, quantum, and autoencoder
capabilities.

This file is importable as `SelfAttentionGen010` from any package
containing the `SelfAttentionGen010.py` module.  The class exposes a
`run` method that accepts input data, rotation and entanglement
parameters, and returns the attended representation.

The design deliberately mirrors the seed `SelfAttention.py` while
adding:
* an optional autoencoder for dimensionality reduction (imported from
  the sibling Autoencoder module);
* a quantum‑derived attention weight generator based on a
  `SamplerQNN` circuit;
* a flag to toggle between fast classical attention and
  quantum‑enhanced attention.

Author: GPT‑OSS‑20B
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import the autoencoder helper from the same package
try:
    from.Autoencoder import Autoencoder, AutoencoderConfig
except Exception:  # pragma: no cover
    # When run outside the package, provide a minimal stub
    Autoencoder = None
    AutoencoderConfig = None

class SelfAttentionGen010(nn.Module):
    """
    Hybrid self‑attention layer.

    Parameters
    ----------
    embed_dim : int
        Dimensionality of the input features.
    use_quantum : bool, default False
        Whether to compute attention weights via a quantum SamplerQNN
        (requires a Qiskit backend).  When False, a classical softmax
        is used.
    autoencoder_cfg : AutoencoderConfig | None, default None
        Configuration for an optional autoencoder.  If provided, the
        input is first compressed by the autoencoder before the
        attention operation.
    """

    def __init__(self,
                 embed_dim: int,
                 *,
                 use_quantum: bool = False,
                 autoencoder_cfg: AutoencoderConfig | None = None) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.use_quantum = use_quantum

        # Optional autoencoder
        self.autoencoder = Autoencoder(embed_dim, **autoencoder_cfg.__dict__) if autoencoder_cfg else None

        # Linear projections for query/key/value (classical)
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj   = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)

        # Quantum‑derived attention weight generator placeholder.
        # In the quantum branch the `run_quantum` method is invoked.
        if use_quantum:
            # The quantum branch uses a SamplerQNN circuit externally;
            # here we keep a reference to the sampler object when set up.
            self.quantum_sampler = None

        # Register parameters to mirror the seed API
        self.register_buffer("rotation_params", torch.zeros(embed_dim * 3))
        self.register_buffer("entangle_params", torch.zeros(embed_dim - 1))

    # ------------------------------------------------------------------
    # Classical attention
    # ------------------------------------------------------------------
    def _classical_attention(self,
                             inputs: torch.Tensor,
                             rotation_params: torch.Tensor,
                             entangle_params: torch.Tensor) -> torch.Tensor:
        """
        Computes standard scaled‑dot‑product attention.

        Parameters
        ----------
        inputs : torch.Tensor
            Input tensor of shape (batch, seq_len, embed_dim).
        rotation_params : torch.Tensor
            Dummy tensor to keep API compatibility; not used.
        entangle_params : torch.Tensor
            Dummy tensor to keep API compatibility; not used.

        Returns
        -------
        torch.Tensor
            Attended representation of shape (batch, seq_len, embed_dim).
        """
        query = self.query_proj(inputs)
        key   = self.key_proj(inputs)
        value = self.value_proj(inputs)

        scores = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(self.embed_dim)
        attn_weights = F.softmax(scores, dim=-1)
        return torch.matmul(attn_weights, value)

    # ------------------------------------------------------------------
    # Quantum attention
    # ------------------------------------------------------------------
    def _quantum_attention(self,
                           inputs: torch.Tensor,
                           rotation_params: torch.Tensor,
                           entangle_params: torch.Tensor,
                           backend) -> torch.Tensor:
        """
        Uses a SamplerQNN to produce attention weights for each feature
        dimension.  The circuit is built externally and the sampler
        returns a probability distribution over the `embed_dim` qubits.

        Parameters
        ----------
        inputs : torch.Tensor
            Input tensor of shape (batch, seq_len, embed_dim).
        rotation_params : torch.Tensor
            Parameters for the rotation part of the circuit.
        entangle_params : torch.Tensor
            Parameters for the entangling part of the circuit.
        backend : qiskit.providers.BaseBackend
            Qiskit backend on which to execute the sampler.

        Returns
        -------
        torch.Tensor
            Attended representation of shape (batch, seq_len, embed_dim).
        """
        if self.quantum_sampler is None:
            raise RuntimeError("Quantum sampler not initialized. "
                               "Call `setup_quantum` with a SamplerQNN instance.")

        # Execute the sampler to get a probability distribution
        # over the `embed_dim` qubits.  The sampler returns a dict
        # mapping bit-strings to counts; we convert to a float array.
        counts = self.quantum_sampler.run(
            backend=backend,
            rotation_params=rotation_params.numpy(),
            entangle_params=entangle_params.numpy(),
            shots=1024,
        )
        probs = np.zeros(self.embed_dim)
        for bitstring, count in counts.items():
            idx = int(bitstring, 2)
            probs[idx] = count
        probs = probs / probs.sum()
        attn_weights = torch.from_numpy(probs).float().unsqueeze(0).unsqueeze(0)
        # Broadcast to match (batch, seq_len, embed_dim)
        attn_weights = attn_weights.repeat(inputs.shape[0], inputs.shape[1], 1)
        return torch.matmul(attn_weights, self.value_proj(inputs))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self,
            inputs: torch.Tensor,
            rotation_params: torch.Tensor,
            entangle_params: torch.Tensor,
            *,
            backend=None) -> torch.Tensor:
        """
        Forward pass compatible with the original seed API.

        Parameters
        ----------
        inputs : torch.Tensor
            Input tensor of shape (batch, seq_len, embed_dim).
        rotation_params : torch.Tensor
            Parameters for rotations; shape depends on the chosen
            branch.
        entangle_params : torch.Tensor
            Parameters for entanglement; shape depends on the chosen
            branch.
        backend : qiskit.providers.BaseBackend, optional
            Required only if `use_quantum=True`.

        Returns
        -------
        torch.Tensor
            The attended representation.
        """
        if self.autoencoder is not None:
            inputs = self.autoencoder(inputs)

        if self.use_quantum:
            if backend is None:
                raise ValueError("Quantum backend must be provided when "
                                 "use_quantum=True.")
            return self._quantum_attention(inputs, rotation_params,
                                           entangle_params, backend)
        else:
            return self._classical_attention(inputs, rotation_params,
                                             entangle_params)

    # ------------------------------------------------------------------
    # Quantum sampler setup helper
    # ------------------------------------------------------------------
    def setup_quantum(self, sampler_qnn):
        """
        Attach a pre‑built SamplerQNN instance to the module.

        Parameters
        ----------
        sampler_qnn : qiskit_machine_learning.neural_networks.SamplerQNN
            Pre‑configured SamplerQNN circuit.
        """
        self.quantum_sampler = sampler_qnn

    def __repr__(self) -> str:
        return (f"SelfAttentionGen010(embed_dim={self.embed_dim}, "
                f"use_quantum={self.use_quantum}, "
                f"autoencoder={self.autoencoder is not None})")

__all__ = ["SelfAttentionGen010"]
