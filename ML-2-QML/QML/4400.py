from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit import QuantumCircuit
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit.primitives import StatevectorSampler

def HybridSamplerQNN():
    """Quantum sampler that combines a Z‑feature map, self‑attention entanglement, and a convolution‑style ansatz."""
    # Two input parameters
    inputs = ParameterVector("input", 2)

    # Weight parameters: 5 for self‑attention (3 rotations + 1 entangle + 1 extra), 10 for two convolution blocks
    weights = ParameterVector("weight", 15)

    # Feature map that embeds the classical inputs into a quantum state
    feature_map = ZFeatureMap(2)

    # Self‑attention entanglement block
    def attention_block(qc: QuantumCircuit, params: ParameterVector) -> QuantumCircuit:
        for i in range(2):
            qc.rx(params[3 * i], i)
            qc.ry(params[3 * i + 1], i)
            qc.rz(params[3 * i + 2], i)
        qc.crx(params[4], 0, 1)
        return qc

    # Simple 2‑qubit convolution block
    def conv_block(qc: QuantumCircuit, params: ParameterVector, idx: int) -> QuantumCircuit:
        qc.cx(0, 1)
        qc.ry(params[idx], 0)
        qc.ry(params[idx + 1], 1)
        return qc

    # Build the ansatz
    ansatz = QuantumCircuit(2)
    ansatz = attention_block(ansatz, weights[0:5])
    ansatz = conv_block(ansatz, weights, 5)
    ansatz = conv_block(ansatz, weights, 7)
    # No explicit measurement – the sampler will infer probabilities

    # Combine feature map and ansatz
    circuit = QuantumCircuit(2)
    circuit.compose(feature_map, range(2), inplace=True)
    circuit.compose(ansatz, range(2), inplace=True)

    sampler = StatevectorSampler()
    sampler_qnn = SamplerQNN(circuit=circuit,
                            input_params=inputs,
                            weight_params=weights,
                            sampler=sampler)
    return sampler_qnn

__all__ = ["HybridSamplerQNN"]
