import torch
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.neural_networks import SamplerQNN

class HybridSamplerQNN:
    """
    Quantum sampler that embeds QCNN‑style convolution layers.
    The circuit consists of a 2‑qubit ZFeatureMap followed by two
    parameterised convolution blocks.  The sampler outputs a probability
    distribution over the four computational basis states, enabling
    hybrid training with the classical counterpart.
    """

    def __init__(self) -> None:
        # Feature (input) parameters: one per qubit
        self.input_params = ParameterVector("x", length=2)
        # Weight (trainable) parameters: two conv blocks, 4 parameters each
        self.weight_params = ParameterVector("w", length=8)

        # Build the quantum circuit
        self.circuit = self._build_circuit()

        # Quantum sampler backend
        self.sampler = StatevectorSampler()

        # Wrap in the Qiskit Machine Learning SamplerQNN
        self.qnn = SamplerQNN(
            circuit=self.circuit,
            input_params=self.input_params,
            weight_params=self.weight_params,
            sampler=self.sampler,
        )

    def _build_circuit(self) -> QuantumCircuit:
        """
        Constructs a 2‑qubit circuit with a ZFeatureMap followed by two
        convolution layers, each parameterised by 4 angles.
        """
        feature_map = ZFeatureMap(2)

        def conv_circuit(params: ParameterVector) -> QuantumCircuit:
            qc = QuantumCircuit(2)
            qc.rz(params[0], 0)
            qc.ry(params[1], 1)
            qc.cx(0, 1)
            qc.ry(params[2], 0)
            qc.rz(params[3], 1)
            return qc

        # Create the circuit
        qc = QuantumCircuit(2)
        qc.compose(feature_map, [0, 1], inplace=True)
        qc.compose(conv_circuit(self.weight_params[0:4]), [0, 1], inplace=True)
        qc.compose(conv_circuit(self.weight_params[4:8]), [0, 1], inplace=True)
        return qc

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that takes a batch of 2‑dimensional input vectors,
        evaluates the quantum sampler, and returns a tensor of shape
        (batch, 4) containing the probability distribution over
        |00>, |01>, |10>, |11>.
        """
        import numpy as np
        # Convert to NumPy for the backend
        inputs_np = inputs.detach().cpu().numpy()
        # The SamplerQNN predict method expects a 2‑D array of shape (batch, 2)
        probs = self.qnn.predict(inputs_np)
        # Convert to a torch tensor
        return torch.tensor(probs, dtype=torch.float32)

__all__ = ["HybridSamplerQNN"]
