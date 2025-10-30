"""Quantum hybrid sampler that mirrors the classical HybridSampler but uses Qiskit circuits."""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector
from qiskit.primitives import Sampler as QuantumSampler
from qiskit_machine_learning.neural_networks import SamplerQNN as QSamplerQNN

# Local imports from the seed modules
from.Conv import Conv
from.FCL import FCL
from.FastBaseEstimator import FastBaseEstimator


class QuantumHybridSampler:
    """
    A quantum‑classical hybrid sampler that:
    * Runs a 2‑qubit convolutional circuit (QuanvCircuit).
    * Runs a 2‑qubit sampler circuit (SamplerQNN).
    * Runs a 1‑qubit fully‑connected circuit (FCLQuantum).
    * Concatenates the scalar outputs into a 3‑element vector.
    * Applies a classical softmax classifier to produce two output probabilities.

    The :meth:`evaluate` method supports both exact (Statevector) and noisy (Sampler) expectation
    calculations, mirroring the FastEstimator interface.
    """

    def __init__(self, simulator: qiskit.providers.BaseBackend | None = None) -> None:
        self.backend = simulator or qiskit.Aer.get_backend("qasm_simulator")
        # Instantiate the three sub‑circuits
        self.conv = Conv()
        self.sampler = QSamplerQNN(
            circuit=QuantumCircuit(2),
            input_params=ParameterVector("input", 2),
            weight_params=ParameterVector("weight", 4),
            sampler=QuantumSampler(self.backend),
        )
        self.fcl = FCL()  # uses its own quantum circuit internally

        # Classical linear classifier
        self.classifier_weights = np.random.randn(3, 2)
        self.classifier_bias = np.random.randn(2)

    def _expectation_statevector(self, circuit: QuantumCircuit, params: list[float]) -> float:
        """Compute exact expectation value with Statevector."""
        bound_circuit = circuit.assign_parameters(dict(zip(circuit.parameters, params)))
        state = Statevector.from_instruction(bound_circuit)
        # Expectation of Z (probability of |1> - |0>)
        return state.expectation_value(qiskit.quantum_info.Pauli("Z")).real

    def evaluate(
        self,
        parameter_sets: list[list[float]],
        shots: int | None = None,
        seed: int | None = None,
    ) -> list[list[float]]:
        """
        Evaluate the hybrid sampler over batches of parameter sets.

        Parameters
        ----------
        parameter_sets : list[list[float]]
            Each inner list concatenates the parameters for the sampler
            (4 values), the FCL (1 value), and the convolution (none).
        shots : int | None, optional
            If provided, uses a noisy Sampler for the sampler circuit.
        seed : int | None, optional
            Random seed for noisy sampler.

        Returns
        -------
        list[list[float]]
            Softmax probabilities for each parameter set.
        """
        rng = np.random.default_rng(seed)
        results: list[list[float]] = []

        for params in parameter_sets:
            # Split parameters
            sampler_params = params[:4]
            fcl_params = params[4:5]

            # 1. Convolutional output (classical)
            conv_out = self.conv.run([])  # Conv has no parameters; returns a scalar

            # 2. Sampler quantum output
            if shots is None:
                sampler_out = self.sampler.circuit.assign_parameters(
                    dict(zip(self.sampler.circuit.parameters, sampler_params))
                )
                state = Statevector.from_instruction(sampler_out)
                sampler_prob = state.expectation_value(qiskit.quantum_info.Pauli("Z")).real
            else:
                sampler_circuit = self.sampler.circuit.assign_parameters(
                    dict(zip(self.sampler.circuit.parameters, sampler_params))
                )
                sampler_out = self.sampler.sampler.run(
                    sampler_circuit,
                    shots=shots,
                    parameter_binds=[{p: v} for p, v in zip(sampler_circuit.parameters, sampler_params)],
                )
                counts = sampler_out.get_counts()
                ones = sum(int(bit) for key in counts for bit in key)
                sampler_prob = ones / (shots * 2)  # two qubits

            # 3. Fully‑connected quantum output
            fcl_out = self.fcl.run(fcl_params)

            # 4. Concatenate and classify
            vec = np.array([conv_out, sampler_prob, fcl_out], dtype=np.float32)
            logits = vec @ self.classifier_weights + self.classifier_bias
            probs = np.exp(logits) / np.sum(np.exp(logits))
            results.append(probs.tolist())

        return results


__all__ = ["QuantumHybridSampler"]
