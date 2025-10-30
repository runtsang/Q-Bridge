import pennylane as qml
import torch


def build_quantum_decoder(num_latent: int, output_dim: int):
    """Return a Pennylane QNode that decodes a latent vector into an output vector.

    The decoder uses a simple variational circuit: each latent qubit is
    encoded via an RX rotation, followed by a single layer of
    StronglyEntanglingLayers.  The circuit outputs the expectation
    value of PauliZ on each wire, which forms the reconstructed
    output.

    Parameters
    ----------
    num_latent : int
        Number of latent qubits (size of the input vector).
    output_dim : int
        Dimension of the reconstructed output (number of qubits used for output).

    Returns
    -------
    decoder_qnode : Callable
        A Pennylane QNode that accepts a torch.Tensor of shape
        (num_latent,) and returns a torch.Tensor of shape
        (output_dim,).
    trainable_params : list
        List of torch.nn.Parameter objects that are the trainable
        parameters of the QNode.
    """
    dev = qml.device("default.qubit", wires=output_dim)

    # Initialize trainable variational parameters for the entangling layer
    # Use a single layer of StronglyEntanglingLayers
    weights = torch.randn(1, output_dim, 3, requires_grad=True)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def decoder_qnode(z: torch.Tensor) -> torch.Tensor:
        # Encode the latent vector as rotations on the first num_latent qubits
        for i in range(num_latent):
            qml.RX(z[i], wires=i)
        # Entanglement layer
        qml.templates.StronglyEntanglingLayers(weights, wires=range(output_dim))
        # Measure expectation values of PauliZ on each wire
        return torch.stack([qml.expval(qml.PauliZ(i)) for i in range(output_dim)], dim=-1)

    return decoder_qnode, [weights]


__all__ = ["build_quantum_decoder"]
