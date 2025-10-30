"""Enhanced quantum sampler network with 3 qubits and a 4-layer variational ansatz."""
from qiskit.circuit import ParameterVector, QuantumCircuit
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit.primitives import StatevectorSampler as Sampler

def EnhancedSamplerQNN() -> SamplerQNN:
    """
    Build a 3-qubit variational sampler:
      * Feature map: 3 Ry gates encoding the 3 input parameters.
      * Ansatz: 4 layers of Ry rotations + CZ entanglement.
      * Uses a statevector sampler for exact sampling.
    """
    # Input parameters
    inputs = ParameterVector("input", 3)
    # Weight parameters: 4 layers * 3 qubits = 12
    weights = ParameterVector("weight", 12)

    qc = QuantumCircuit(3)

    # Feature map: encode inputs
    for i in range(3):
        qc.ry(inputs[i], i)

    # Variational ansatz
    for layer in range(4):
        for i in range(3):
            qc.ry(weights[layer * 3 + i], i)
        # Entanglement pattern
        qc.cz(0, 1)
        qc.cz(1, 2)
        qc.cz(2, 0)

    # No measurement; SamplerQNN will handle sampling
    sampler = Sampler()
    sampler_qnn = SamplerQNN(
        circuit=qc,
        input_params=inputs,
        weight_params=weights,
        sampler=sampler,
    )
    return sampler_qnn
