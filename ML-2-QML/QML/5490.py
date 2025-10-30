import numpy as np
import qiskit
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import ParameterVector
from typing import Iterable, Tuple, Optional

class FCLQuantum:
    """
    Quantum counterpart to the hybrid FCL module.
    Combines data encoding, variational layers, a convolution‑style
    sub‑circuit, and measurement of Z observables that produce a
    decision vector.
    """

    def __init__(self,
                 num_qubits: int,
                 depth: int,
                 shots: int = 1024,
                 backend: Optional[qiskit.providers.Backend] = None) -> None:
        self.num_qubits = num_qubits
        self.depth = depth
        self.shots = shots
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.circuit, self.encoding, self.weights, self.observables = self._build_circuit()

    def _build_circuit(self) -> Tuple[QuantumCircuit, Iterable, Iterable, list]:
        # Data encoding with RX rotations
        encoding = ParameterVector("x", self.num_qubits)
        # Variational parameters
        weights = ParameterVector("theta", self.num_qubits * self.depth)
        circuit = QuantumCircuit(self.num_qubits)

        for param, qubit in zip(encoding, range(self.num_qubits)):
            circuit.rx(param, qubit)

        idx = 0
        for _ in range(self.depth):
            for qubit in range(self.num_qubits):
                circuit.ry(weights[idx], qubit)
                idx += 1
            for qubit in range(self.num_qubits - 1):
                circuit.cz(qubit, qubit + 1)

        # Convolution‑style sub‑circuit (4‑qubit random pattern)
        if self.num_qubits >= 4:
            conv = QuantumCircuit(4)
            conv.h(range(4))
            conv.cx(0, 1)
            conv.cx(2, 3)
            conv.barrier()
            conv.cx(1, 2)
            conv.cx(3, 0)
            circuit.append(conv, range(4))

        # Measurement of Z observables
        observables = [f"Z{i}" for i in range(self.num_qubits)]
        return circuit, encoding, weights, observables

    def run(self,
            data: Iterable[float],
            thetas: Iterable[float]) -> np.ndarray:
        """
        Execute the circuit with given data and variational parameters.

        Parameters
        ----------
        data : Iterable[float]
            Input feature vector of length ``num_qubits``.
        thetas : Iterable[float]
            Flattened list of variational parameters of length ``num_qubits * depth``.

        Returns
        -------
        np.ndarray
            Expectation value of each Z observable, shape (num_qubits,).
        """
        if len(data)!= self.num_qubits:
            raise ValueError(f"Data length {len(data)} must equal num_qubits {self.num_qubits}")
        if len(thetas)!= self.num_qubits * self.depth:
            raise ValueError(f"Expected {self.num_qubits * self.depth} theta parameters, got {len(thetas)}")

        param_binds = {str(self.encoding[i]): data[i] for i in range(self.num_qubits)}
        param_binds.update({str(self.weights[i]): thetas[i] for i in range(len(thetas))})
        job = execute(self.circuit,
                      self.backend,
                      shots=self.shots,
                      parameter_binds=[param_binds])
        result = job.result()
        counts = result.get_counts(self.circuit)

        exp_vals = []
        for qubit in range(self.num_qubits):
            exp = 0.0
            for bitstring, count in counts.items():
                # Qiskit returns bitstrings MSB first
                bit = int(bitstring[self.num_qubits - qubit - 1])
                exp += (1 - 2 * bit) * count
            exp /= self.shots
            exp_vals.append(exp)
        return np.array(exp_vals)

    def run_expectation(self,
                        data: Iterable[float],
                        thetas: Iterable[float]) -> np.ndarray:
        """
        Convenience wrapper that returns a single expectation vector.
        """
        return self.run(data, thetas)


__all__ = ["FCLQuantum"]
