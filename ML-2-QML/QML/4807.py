"""
Quantum counterpart of SamplerQNNGen261.  Uses Qiskit to implement:
1. A random‑circuit convolution (QuanvCircuit) that thresholds classical data.
2. A photonic‑style gate sequence (FraudCircuit) built with standard Qiskit gates.
3. A parameterised sampler circuit (SamplerCircuit) that outputs a 2‑class probability.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import ParameterVector, Parameter
from qiskit.circuit.random import random_circuit
from qiskit.primitives import Sampler as StatevectorSampler
from dataclasses import dataclass
from typing import Iterable, Tuple, List, Optional


# ----- Convolution circuit -----------------------------------------------
class QuanvCircuit:
    """
    Quantum filter that emulates a convolutional layer.
    Each qubit corresponds to one pixel; data values threshold the rotation angles.
    """

    def __init__(self, kernel_size: int, backend, shots: int, threshold: float) -> None:
        self.n_qubits = kernel_size ** 2
        self.backend = backend
        self.shots = shots
        self.threshold = threshold
        self.theta = [Parameter(f"θ{i}") for i in range(self.n_qubits)]
        self.circuit = QuantumCircuit(self.n_qubits)
        for i in range(self.n_qubits):
            self.circuit.rx(self.theta[i], i)
        self.circuit.barrier()
        self.circuit += random_circuit(self.n_qubits, depth=2, skip_barriers=True)
        self.circuit.measure_all()

    def run(self, data: np.ndarray) -> float:
        """
        Execute the circuit on a batch of data and return the average
        probability of measuring |1> across all qubits.

        Parameters
        ----------
        data : np.ndarray
            2‑D array with shape (kernel_size, kernel_size).

        Returns
        -------
        float
            Normalised mean |1> probability.
        """
        flat = np.reshape(data, (1, self.n_qubits))
        param_binds = []
        for row in flat:
            bind = {}
            for i, val in enumerate(row):
                bind[self.theta[i]] = np.pi if val > self.threshold else 0.0
            param_binds.append(bind)

        job = execute(
            self.circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=param_binds,
        )
        result = job.result().get_counts(self.circuit)
        total_counts = sum(result.values())
        ones = sum(
            sum(int(bit) for bit in key) * count for key, count in result.items()
        )
        return ones / (self.shots * self.n_qubits)


# ----- Photonic‑style fraud circuit ---------------------------------------
@dataclass
class FraudLayerParameters:
    bs_theta: float
    bs_phi: float
    phases: Tuple[float, float]
    squeeze_r: Tuple[float, float]
    squeeze_phi: Tuple[float, float]
    displacement_r: Tuple[float, float]
    displacement_phi: Tuple[float, float]
    kerr: Tuple[float, float]


def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))


class FraudCircuit:
    """
    Builds a gate sequence that imitates photonic operations using
    standard Qiskit gates.  Two qubits model the two optical modes.
    """

    def __init__(self, params: FraudLayerParameters, clip: bool = True) -> None:
        self.circuit = QuantumCircuit(2)
        # Beam splitter mimicked by a CZ and RZ
        self.circuit.cz(0, 1)
        self.circuit.rz(params.bs_theta, 0)
        self.circuit.rz(params.bs_phi, 1)
        for i, phase in enumerate(params.phases):
            self.circuit.rz(phase, i)
        for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
            self.circuit.rx(_clip(r, 5.0) if clip else r, i)
            self.circuit.rz(_clip(phi, 5.0) if clip else phi, i)
        # Second beam splitter
        self.circuit.cz(0, 1)
        for i, phase in enumerate(params.phases):
            self.circuit.rz(phase, i)
        for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
            self.circuit.rx(_clip(r, 5.0) if clip else r, i)
            self.circuit.rz(_clip(phi, 5.0) if clip else phi, i)
        for i, k in enumerate(params.kerr):
            self.circuit.rx(_clip(k, 1.0) if clip else k, i)

    def run(self, data: float) -> float:
        """
        Execute the circuit with the data value encoded as an angle
        on qubit 0.  Returns the probability of measuring |1> on qubit 0.

        Parameters
        ----------
        data : float
            Scalar derived from previous layer output.

        Returns
        -------
        float
            Probability of |1> on qubit 0.
        """
        bind = {self.circuit.parameters[0]: data}
        job = execute(
            self.circuit,
            Aer.get_backend("qasm_simulator"),
            shots=1024,
            parameter_binds=[bind],
        )
        counts = job.result().get_counts(self.circuit)
        return counts.get("1", 0) / 1024


# ----- Sampler circuit -----------------------------------------------
class SamplerCircuit:
    """
    Variational sampler that outputs a 2‑class probability distribution.
    """

    def __init__(self) -> None:
        self.input_params = ParameterVector("input", 2)
        self.weight_params = ParameterVector("weight", 4)
        self.circuit = QuantumCircuit(2)
        self.circuit.ry(self.input_params[0], 0)
        self.circuit.ry(self.input_params[1], 1)
        self.circuit.cx(0, 1)
        self.circuit.ry(self.weight_params[0], 0)
        self.circuit.ry(self.weight_params[1], 1)
        self.circuit.cx(0, 1)
        self.circuit.ry(self.weight_params[2], 0)
        self.circuit.ry(self.weight_params[3], 1)
        self.circuit.measure_all()

    def run(self, probs: Tuple[float, float]) -> Tuple[float, float]:
        """
        Execute the sampler circuit with the supplied probabilities as
        input parameters and return the resulting 2‑class distribution.

        Parameters
        ----------
        probs : Tuple[float, float]
            Probabilities to encode as rotation angles.

        Returns
        -------
        Tuple[float, float]
            Normalised probability distribution over two outcomes.
        """
        bind = {
            self.input_params[0]: probs[0],
            self.input_params[1]: probs[1],
        }
        # Use a state‑vector sampler for exact probabilities
        sampler = StatevectorSampler()
        result = sampler.run(self.circuit, parameter_binds=[bind])
        probs_out = result.get_probabilities()
        return (probs_out[0], probs_out[1])


# ----- Hybrid QML module -----------------------------------------------
class SamplerQNNGen261:
    """
    Quantum analogue of the classical SamplerQNNGen261.
    Processes a 2×2 input image through a quantum convolution, a
    photonic‑style fraud circuit, and a sampler circuit.
    """

    def __init__(
        self,
        conv_size: int = 2,
        conv_threshold: float = 127,
        fraud_params: Optional[FraudLayerParameters] = None,
    ) -> None:
        backend = Aer.get_backend("qasm_simulator")
        self.conv = QuanvCircuit(conv_size, backend, shots=100, threshold=conv_threshold)

        if fraud_params is None:
            fraud_params = FraudLayerParameters(
                bs_theta=0.5,
                bs_phi=0.5,
                phases=(0.0, 0.0),
                squeeze_r=(0.1, 0.1),
                squeeze_phi=(0.0, 0.0),
                displacement_r=(0.05, 0.05),
                displacement_phi=(0.0, 0.0),
                kerr=(0.0, 0.0),
            )
        self.fraud = FraudCircuit(fraud_params, clip=True)
        self.sampler = SamplerCircuit()

    def run(self, image: np.ndarray) -> Tuple[float, float]:
        """
        Execute the full hybrid pipeline.

        Parameters
        ----------
        image : np.ndarray
            2‑D array of shape (2, 2) with integer pixel values.

        Returns
        -------
        Tuple[float, float]
            Final output probabilities from the sampler circuit.
        """
        conv_prob = self.conv.run(image)                     # Step 1
        fraud_prob = self.fraud.run(conv_prob)               # Step 2
        sampler_out = self.sampler.run((conv_prob, fraud_prob))  # Step 3
        return sampler_out


__all__ = ["SamplerQNNGen261"]
