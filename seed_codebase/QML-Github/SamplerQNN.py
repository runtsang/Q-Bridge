from qiskit.circuit import ParameterVector

def SamplerQNN():
    """A simple example of a parameterized quantum circuit for a SamplerQNN."""
    inputs2 = ParameterVector("input", 2)
    weights2 = ParameterVector("weight", 4)
    print(f"input parameters: {[str(item) for item in inputs2.params]}")
    print(f"weight parameters: {[str(item) for item in weights2.params]}")

    qc2 = QuantumCircuit(2)
    qc2.ry(inputs2[0], 0)
    qc2.ry(inputs2[1], 1)
    qc2.cx(0, 1)
    qc2.ry(weights2[0], 0)
    qc2.ry(weights2[1], 1)
    qc2.cx(0, 1)
    qc2.ry(weights2[2], 0)
    qc2.ry(weights2[3], 1)

    qc2.draw("mpl", style="clifford")

    from qiskit_machine_learning.neural_networks import SamplerQNN
    from qiskit.primitives import StatevectorSampler as Sampler

    sampler = Sampler()
    sampler_qnn = SamplerQNN(circuit=qc2, input_params=inputs2, weight_params=weights2, sampler=sampler)
    return sampler_qnn