"""Quantum autoencoder using Qiskit with feature map and measurement-based decoder."""
from qiskit import QuantumCircuit, ClassicalRegister
from qiskit.circuit import Parameter
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Sampler
from qiskit_machine_learning.neural_networks import SamplerQNN
import numpy as np

def Autoencoder__gen300():
    """
    Returns a SamplerQNN that encodes classical data into a quantum state via a feature map,
    processes it with a variational ansatz, and decodes by measuring all qubits.
    """
    num_qubits = 4
    reps = 2

    # Variational ansatz
    ansatz = RealAmplitudes(num_qubits, reps=reps)

    # Sampler primitive
    sampler = Sampler()

    # Input parameters for feature map
    input_params = [Parameter(f"x{i}") for i in range(num_qubits)]

    def circuit(inputs, weights):
        """Builds a circuit for given inputs and variational weights."""
        qc = QuantumCircuit(num_qubits)

        # Feature map: simple RX/RZ rotations encoding each input value
        for idx, val in enumerate(inputs):
            qc.rx(val, idx)
            qc.rz(val, idx)

        # Variational part
        qc.compose(ansatz.bind_parameters(weights), inplace=True)

        # Measurement-based decoder: measure all qubits
        cr = ClassicalRegister(num_qubits)
        qc.add_register(cr)
        qc.measure(range(num_qubits), cr)
        return qc

    def interpret(result):
        """
        Convert a list of Result objects into a probability vector over bitstrings.
        """
        probs = np.zeros(2**num_qubits)
        for r in result:
            counts = r.get_counts()
            for bitstring, cnt in counts.items():
                idx = int(bitstring, 2)
                probs[idx] += cnt
        probs /= probs.sum()
        return probs

    qnn = SamplerQNN(
        circuit=circuit,
        input_params=input_params,
        weight_params=ansatz.parameters,
        interpret=interpret,
        output_shape=2**num_qubits,
        sampler=sampler,
    )
    return qnn
