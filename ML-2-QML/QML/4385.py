from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit.primitives import Sampler as StatevectorSampler

def SamplerQNNGen062():
    """Quantum sampler that integrates kernel, attention and fully‑connected logic."""
    # Input and weight parameters
    inputs = ParameterVector("input", 2)
    weights_kernel = ParameterVector("w_kernel", 4)
    weights_attn = ParameterVector("w_attn", 4)
    weights_fc = ParameterVector("w_fc", 4)

    qc = QuantumCircuit(4)
    # Input encoding
    qc.ry(inputs[0], 0)
    qc.ry(inputs[1], 1)
    # Kernel layer
    qc.ry(weights_kernel[0], 0)
    qc.ry(weights_kernel[1], 1)
    qc.cx(0, 2)
    qc.cx(1, 3)
    # Attention layer
    qc.ry(weights_attn[0], 2)
    qc.ry(weights_attn[1], 3)
    qc.crx(weights_attn[2], 0, 1)
    qc.crx(weights_attn[3], 2, 3)
    # Fully‑connected layer
    qc.ry(weights_fc[0], 0)
    qc.ry(weights_fc[1], 1)
    qc.ry(weights_fc[2], 2)
    qc.ry(weights_fc[3], 3)

    sampler = StatevectorSampler()
    sampler_qnn = SamplerQNN(
        circuit=qc,
        input_params=inputs,
        weight_params=weights_kernel + weights_attn + weights_fc,
        sampler=sampler,
    )
    return sampler_qnn

__all__ = ["SamplerQNNGen062"]
