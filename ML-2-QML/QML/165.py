import numpy as np
import qiskit
from qiskit import transpile, assemble

class QuantumHybridBinaryClassifier:
    """
    Quantum‑only component that produces a probability from a
    two‑qubit variational circuit.  It can be used directly
    in a hybrid pipeline or as a stand‑alone classifier when
    the classical feature extractor is omitted.
    """
    def __init__(self, backend=None, shots=1024):
        self.backend = backend or qiskit.Aer.get_backend("aer_simulator")
        self.shots = shots
        self.circuit = qiskit.QuantumCircuit(2)
        self.theta = qiskit.circuit.Parameter("theta")
        self.circuit.h([0, 1])
        self.circuit.ry(self.theta, [0, 1])
        self.circuit.measure_all()

    def predict(self, inputs):
        """
        Compute the probability for each input parameter set.
        :param inputs: numpy array of shape (batch, 2)
        :return: numpy array of shape (batch,) containing probabilities
        """
        exp_vals = self._run(inputs)
        probs = 1 / (1 + np.exp(-exp_vals))
        return probs

    def _run(self, params):
        if params.ndim == 1:
            params = params.reshape(1, -1)
        compiled = transpile(self.circuit, self.backend)
        param_binds = [{self.theta: [p[0], p[1]]} for p in params]
        qobj = assemble(compiled, shots=self.shots, parameter_binds=param_binds)
        job = self.backend.run(qobj)
        result = job.result()
        exp_vals = []
        for counts in result.get_counts():
            exp = 0.0
            for bitstring, cnt in counts.items():
                exp += int(bitstring, 2) * cnt
            exp /= self.shots
            exp_vals.append(exp)
        return np.array(exp_vals)

__all__ = ["QuantumHybridBinaryClassifier"]
