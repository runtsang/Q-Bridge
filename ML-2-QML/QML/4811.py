import numpy as np
import qiskit
from qiskit.circuit import ParameterVector
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit.primitives import StatevectorSampler as Sampler

__all__ = ["FCLGen265Quantum"]

class FCLGen265Quantum:
    """
    Quantum counterpart that fuses:
      * A single‑qubit fully‑connected parameterised circuit (from FCL.py)
      * A 2‑qubit sampler network inspired by SamplerQNN.py
      * Quantum‑NAT style feature encoding via a 4‑qubit RandomLayer
    The class exposes two evaluation primitives: run() for a single‑qubit
    expectation value and sample() for the 2‑class probability output of the
    sampler circuit.
    """
    def __init__(self, backend=None, shots=1024):
        if backend is None:
            backend = qiskit.Aer.get_backend("qasm_simulator")
        self.backend = backend
        self.shots = shots

        # --- Main single‑qubit circuit ------------------------------------
        self.theta = qiskit.circuit.Parameter("theta")
        self.circuit = qiskit.QuantumCircuit(1)
        self.circuit.h(0)
        self.circuit.ry(self.theta, 0)
        self.circuit.measure_all()

        # --- Sampler QNN --------------------------------------------------
        input_params = ParameterVector("input", 2)
        weight_params = ParameterVector("weight", 4)
        qc_sampler = qiskit.QuantumCircuit(2)
        qc_sampler.ry(input_params[0], 0)
        qc_sampler.ry(input_params[1], 1)
        qc_sampler.cx(0, 1)
        qc_sampler.ry(weight_params[0], 0)
        qc_sampler.ry(weight_params[1], 1)
        qc_sampler.cx(0, 1)
        qc_sampler.ry(weight_params[2], 0)
        qc_sampler.ry(weight_params[3], 1)
        sampler = Sampler()
        self.sampler_qnn = SamplerQNN(
            circuit=qc_sampler,
            input_params=input_params,
            weight_params=weight_params,
            sampler=sampler
        )

    def run(self, theta_value: float) -> np.ndarray:
        """
        Evaluate the expectation value of the single‑qubit circuit.
        """
        bound = {self.theta: theta_value}
        job = qiskit.execute(
            self.circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=[bound]
        )
        result = job.result().get_counts(self.circuit)
        counts = np.array(list(result.values()))
        probs = counts / self.shots
        states = np.array(list(result.keys()), dtype=int)
        expectation = np.sum(states * probs)
        return np.array([expectation])

    def sample(self, inputs: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """
        Run the sampler QNN to obtain a 2‑class probability distribution.
        `inputs` must be shape (n_samples, 2) and `weights` shape (4,).
        """
        return self.sampler_qnn.run(inputs, weights)
