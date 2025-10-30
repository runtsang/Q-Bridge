import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.primitives import StatevectorSampler

class SamplerQNN:
    """
    Quantum sampler that implements a QCNN‑style convolution+pooling ansatz
    to generate a 2‑class probability distribution from a 2‑qubit system.
    """
    def __init__(self, seed: int | None = None) -> None:
        if seed is not None:
            np.random.seed(seed)

        # Parameter vectors
        self.input_params = ParameterVector("x", 2)
        self.weight_params = ParameterVector("θ", 6)  # 2 conv + 2 pool + 2 final

        # Build the circuit
        self.circuit = self._build_circuit()
        self.sampler = StatevectorSampler()

    def _conv_circuit(self, params: ParameterVector) -> QuantumCircuit:
        """Two‑qubit convolution block."""
        qc = QuantumCircuit(2)
        qc.ry(params[0], 0)
        qc.ry(params[1], 1)
        qc.cx(0, 1)
        return qc

    def _pool_circuit(self, params: ParameterVector) -> QuantumCircuit:
        """Two‑qubit pooling block."""
        qc = QuantumCircuit(2)
        qc.rz(np.pi / 2, 0)
        qc.cx(0, 1)
        qc.rz(params[0], 0)
        qc.ry(params[1], 1)
        qc.cx(1, 0)
        return qc

    def _build_circuit(self) -> QuantumCircuit:
        """Assembles the QCNN‑style sampler circuit."""
        feature_map = ZFeatureMap(2)
        conv = self._conv_circuit(self.weight_params[0:2])
        pool = self._pool_circuit(self.weight_params[2:4])

        final = QuantumCircuit(2)
        final.ry(self.weight_params[4], 0)
        final.ry(self.weight_params[5], 1)

        circuit = QuantumCircuit(2)
        circuit.append(feature_map, [0, 1])
        circuit.append(conv, [0, 1])
        circuit.append(pool, [0, 1])
        circuit.append(final, [0, 1])
        return circuit

    def sample(self, inputs: np.ndarray) -> np.ndarray:
        """
        Sample the circuit with the provided 2‑dimensional input vector.
        Returns a 2‑element array of probabilities for measuring the first
        qubit in state |0⟩ and |1⟩.
        """
        if inputs.ndim == 1:
            inputs = inputs.reshape(1, -1)
        probs = []
        for inp in inputs:
            param_dict = {
                self.input_params[0]: inp[0],
                self.input_params[1]: inp[1],
            }
            bound = self.circuit.bind_parameters(param_dict)
            state = self.sampler.run(bound).result().get_statevector()
            p0 = p1 = 0.0
            for idx, amp in enumerate(state):
                prob = abs(amp) ** 2
                if idx & 1 == 0:  # qubit 0 = 0
                    p0 += prob
                else:
                    p1 += prob
            probs.append([p0, p1])
        return np.array(probs)

__all__ = ["SamplerQNN"]
