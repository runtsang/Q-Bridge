"""Quantum implementation of the hybrid fraud‑detection logic.

The :class:`FraudDetectionHybrid` class encapsulates a variational circuit that
mimics the flow of the classical counterpart:
1. **Input encoding** – angle‑encoding of the feature vector into the first
   ``input_dim`` qubits.
2. **Variational layer** – a RealAmplitudes ansatz with ``latent_dim`` repetitions.
3. **Entanglement block** – a chain of RXX gates parametrised by ``entangle_params``.
4. **Measurement** – probability of the first qubit being in state ``|1⟩`` is used
   as the fraud score.

The circuit is constructed once and parameters are bound at runtime.  The
``run`` method returns a float in ``[0, 1]`` that can be interpreted as a fraud
probability.  The implementation relies solely on Qiskit and NumPy, keeping
the quantum side entirely self‑contained.
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import RealAmplitudes
from qiskit.providers.aer import AerSimulator

class FraudDetectionHybrid:
    """
    Quantum‑variational fraud‑detection circuit that parallels the classical
    :class:`FraudDetectionHybrid` module.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 3,
        num_qubits: int = 4,
        shots: int = 1024,
        backend=None,
    ) -> None:
        """
        Parameters
        ----------
        input_dim : int
            Dimensionality of the classical feature vector.
        latent_dim : int
            Number of variational repetitions (controls circuit depth).
        num_qubits : int
            Total qubit count (input + variational + measurement).
        shots : int
            Number of shots for the backend simulation.
        backend : qiskit.providers.backend.Backend, optional
            Backend to execute the circuit.  Defaults to AerSimulator.
        """
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_qubits = num_qubits
        self.shots = shots
        self.backend = backend or AerSimulator()
        # Parameter vectors for rotation and entanglement
        self.rotation_params = ParameterVector("theta", length=latent_dim * 3)
        self.entangle_params = ParameterVector("phi", length=latent_dim - 1)
        # Build the template circuit once
        self.circuit_template = self._build_template()

    def _build_template(self) -> QuantumCircuit:
        """Creates a parameterised circuit with placeholders for inputs and variational parameters."""
        qr = QuantumRegister(self.num_qubits, "q")
        cr = ClassicalRegister(self.num_qubits, "c")
        qc = QuantumCircuit(qr, cr)

        # 1. Input encoding – angle‑encoding on the first ``input_dim`` qubits
        input_params = ParameterVector("x", length=self.input_dim)
        for i in range(self.input_dim):
            qc.ry(input_params[i], i)

        # 2. Variational RealAmplitudes block
        for i in range(self.latent_dim):
            qc.ry(self.rotation_params[3 * i], i)
            qc.rz(self.rotation_params[3 * i + 1], i)
            qc.rx(self.rotation_params[3 * i + 2], i)

        # 3. Entanglement (RXX chain)
        for i in range(self.latent_dim - 1):
            qc.rxx(self.entangle_params[i], i, i + 1)

        # 4. Measurement of all qubits (output will be read from the first qubit)
        qc.measure_all()
        return qc

    def run(self, inputs: list[float]) -> float:
        """
        Execute the circuit on the provided inputs and return the probability
        of measuring state ``|1⟩`` on the first qubit.

        Parameters
        ----------
        inputs : list[float]
            Feature vector of length ``input_dim``.

        Returns
        -------
        float
            Fraud probability in the interval ``[0, 1]``.
        """
        if len(inputs)!= self.input_dim:
            raise ValueError(f"Expected {self.input_dim} input values, got {len(inputs)}")

        # Bind input parameters
        input_bindings = {f"x{i}": float(val) for i, val in enumerate(inputs)}

        # Bind rotation and entanglement parameters – for a demo we use random values
        rotation_bindings = {
            f"theta{i}": float(np.random.randn()) for i in range(self.rotation_params.length)
        }
        entangle_bindings = {
            f"phi{i}": float(np.random.randn()) for i in range(self.entangle_params.length)
        }

        # Combine all bindings
        param_bindings = {**input_bindings, **rotation_bindings, **entangle_bindings}

        # Create a bound circuit
        bound_circuit = self.circuit_template.bind_parameters(param_bindings)

        # Execute
        job = execute(bound_circuit, self.backend, shots=self.shots)
        result = job.result()
        counts = result.get_counts(bound_circuit)

        # Compute probability of first qubit being in state |1>
        prob_one = sum(counts[state] for state in counts if state[-1] == "1") / self.shots
        return prob_one

    def __call__(self, inputs: list[float]) -> float:
        """Convenience wrapper that forwards to :meth:`run`."""
        return self.run(inputs)

__all__ = ["FraudDetectionHybrid"]
