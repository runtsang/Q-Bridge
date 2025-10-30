from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit.primitives import StatevectorSampler

def HybridSamplerQNN():
    """Quantum sampler network with parameterized circuit, random layer, and entanglement."""
    # Input and weight parameters
    inputs = ParameterVector("input", 2)
    weights = ParameterVector("weight", 4)

    qc = QuantumCircuit(2)
    # Input rotations
    qc.ry(inputs[0], 0)
    qc.ry(inputs[1], 1)
    # Entanglement
    qc.cx(0, 1)
    # Variational layer
    qc.ry(weights[0], 0)
    qc.ry(weights[1], 1)
    qc.cx(0, 1)
    qc.ry(weights[2], 0)
    qc.ry(weights[3], 1)
    # Lightweight random layer (classical angles)
    qc.ry(0.5, 0)
    qc.ry(1.2, 1)

    sampler = StatevectorSampler()
    sampler_qnn = SamplerQNN(circuit=qc,
                             input_params=inputs,
                             weight_params=weights,
                             sampler=sampler)
    return sampler_qnn

__all__ = ["HybridSamplerQNN"]
