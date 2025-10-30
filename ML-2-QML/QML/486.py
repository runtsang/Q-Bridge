import pennylane as qml
import numpy as np

class SamplerQNN:
    """
    A multi‑layer variational quantum sampler.
    Extends the original 2‑qubit circuit with configurable depth,
    parameterized Ry rotations, and a CNOT entanglement chain.
    Provides a QNode that can be used with Pennylane's optimizers
    and a convenient sampling method.
    """
    def __init__(self, n_qubits=2, n_layers=3, device_name='default.qubit', seed=None):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dev = qml.device(device_name, wires=n_qubits)
        if seed is not None:
            np.random.seed(seed)
        self.params = np.random.randn(n_layers, n_qubits) * np.pi

    def _variational_circuit(self, params, inputs):
        # Encode inputs as Ry rotations
        for i in range(self.n_qubits):
            qml.Ry(inputs[i], wires=i)
        # Layered ansatz
        for layer in range(self.n_layers):
            for q in range(self.n_qubits):
                qml.Ry(params[layer, q], wires=q)
            # Entangling chain
            for q in range(self.n_qubits - 1):
                qml.CNOT(wires=[q, q + 1])
            qml.CNOT(wires=[self.n_qubits - 1, 0])  # wrap‑around for full connectivity
        # Measurement
        return [qml.expval(qml.PauliZ(w)) for w in range(self.n_qubits)]

    def get_qnode(self, params=None):
        if params is None:
            params = self.params
        @qml.qnode(self.dev)
        def circuit(inputs):
            return self._variational_circuit(params, inputs)
        return circuit

    def sample(self, shots=1024, params=None):
        """
        Sample measurement outcomes from the variational circuit.
        Returns a dictionary of counts for each bitstring.
        """
        if params is None:
            params = self.params
        circuit = self.get_qnode(params)
        # Use the device's sample method
        result = self.dev.run(circuit, shots=shots)
        counts = {}
        for outcome in result.samples:
            bitstring = ''.join(str(int(bit)) for bit in outcome)
            counts[bitstring] = counts.get(bitstring, 0) + 1
        return counts
