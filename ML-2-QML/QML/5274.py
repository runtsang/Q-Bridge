from qiskit.circuit import ParameterVector
from qiskit import QuantumCircuit
from qiskit_machine_learning.neural_networks import SamplerQNN as QiskitSamplerQNN
from qiskit.primitives import StatevectorSampler as Sampler

def SamplerQNN__gen158():
    """
    Quantum sampler that merges a 2‑qubit feature encoder with a 4‑qubit
    quantum transformer block, providing a simple variational circuit.
    The circuit is sampled using a state‑vector simulator.
    """
    # Parameter vectors
    inputs = ParameterVector("input", 2)
    weights = ParameterVector("weight", 4)

    # 6‑qubit circuit: 2 feature qubits + 4 transformer qubits
    qc = QuantumCircuit(6)

    # Feature encoding on qubits 0 and 1
    qc.ry(inputs[0], 0)
    qc.ry(inputs[1], 1)
    qc.cx(0, 1)

    # Quantum transformer block on qubits 2‑5
    for i in range(4):
        qc.ry(weights[i], i + 2)          # parameterized rotation
    # Entanglement pattern resembling a simple transformer
    qc.cx(2, 3)
    qc.cx(3, 4)
    qc.cx(4, 5)
    qc.cx(5, 2)

    # Measurement on all qubits
    qc.measure_all()

    # Sampler primitive
    sampler = Sampler()
    sampler_qnn = QiskitSamplerQNN(
        circuit=qc,
        input_params=inputs,
        weight_params=weights,
        sampler=sampler,
    )
    return sampler_qnn
