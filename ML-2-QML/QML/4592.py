"""Quantum-centric estimator that can evaluate quantum circuits, quantum neural networks,
and hybrid models such as the Quantum NAT (QFCModel) implemented with torchquantum.

The estimator accepts either a raw Qiskit `QuantumCircuit` or a torchquantum `QuantumModule`
(e.g. QFCModel).  It provides a unified evaluate method and helper to construct a
quantum autoencoder with SamplerQNN.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Union, Any

import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.primitives import Sampler as QiskitSampler
from.QuantumNAT import QFCModel
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import StatevectorSampler as Sampler


class FastHybridEstimator:
    """Hybrid estimator that can evaluate:

    * A pure Qiskit `QuantumCircuit`.
    * A torchquantum `QuantumModule` (e.g. QFCModel), which is a hybrid
      classical‑quantum neural network.
    """

    def __init__(self, model: Union[QuantumCircuit, tq.QuantumModule]) -> None:
        if isinstance(model, QuantumCircuit):
            self._qc = model
            self._params = list(self._qc.parameters)
        elif isinstance(model, tq.QuantumModule):
            self._qmodule = model
        else:
            raise TypeError("model must be a QuantumCircuit or a torchquantum.QuantumModule")

    def evaluate(
        self,
        observables: Iterable,
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[complex]]:
        """Compute expectation values for each parameter set.

        Supports two backends:
        * Qiskit Statevector for a raw circuit.
        * torchquantum simulation for a QuantumModule.
        """
        if hasattr(self, "_qc"):  # Qiskit circuit
            return self._evaluate_circuit(observables, parameter_sets, shots, seed)
        else:
            return self._evaluate_qmodule(observables, parameter_sets, shots, seed)

    def _evaluate_circuit(
        self,
        observables: Iterable["qiskit.quantum_info.operators.base_operator.BaseOperator"],
        parameter_sets: Sequence[Sequence[float]],
        shots: int | None,
        seed: int | None,
    ) -> List[List[complex]]:
        results: List[List[complex]] = []
        for params in parameter_sets:
            if len(params)!= len(self._params):
                raise ValueError("Parameter count mismatch for bound circuit.")
            mapping = dict(zip(self._params, params))
            bound = self._qc.assign_parameters(mapping, inplace=False)
            if shots is None:
                state = Statevector.from_instruction(bound)
                row = [state.expectation_value(obs) for obs in observables]
            else:
                sampler = QiskitSampler(seed=seed)
                result = sampler.run(bound, observables=observables, seed=seed).result()
                # For simplicity we use the noiseless statevector expectation
                state = Statevector.from_instruction(bound)
                row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results

    def _evaluate_qmodule(
        self,
        observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]],
        parameter_sets: Sequence[Sequence[float]],
        shots: int | None,
        seed: int | None,
    ) -> List[List[float]]:
        """Evaluate a torchquantum module.  Observables are callables on the output tensor."""
        # Prepare a dummy input; the module may ignore it
        dummy_input = torch.zeros(1)
        results_list: List[List[float]] = []

        for params in parameter_sets:
            # Set the module parameters
            for p, val in zip(self._qmodule.parameters(), params):
                p.data.copy_(torch.tensor(val, dtype=p.dtype))
            with torch.no_grad():
                output = self._qmodule(dummy_input)
            row: List[float] = []
            for obs in observables:
                val = obs(output)
                if isinstance(val, torch.Tensor):
                    row.append(float(val.mean().cpu()))
                else:
                    row.append(float(val))
            results_list.append(row)
        return results_list

    # Convenience: build a Quantum Autoencoder using SamplerQNN
    @staticmethod
    def build_autoencoder(num_latent: int, num_trash: int) -> SamplerQNN:
        """Return a SamplerQNN representing a simple quantum autoencoder."""
        qr = QuantumCircuit(num_latent + 2 * num_trash + 1, 1)
        ansatz = RealAmplitudes(num_latent + num_trash, reps=5)
        qr.compose(ansatz, range(0, num_latent + num_trash), inplace=True)
        qr.barrier()
        aux = num_latent + 2 * num_trash
        qr.h(aux)
        for i in range(num_trash):
            qr.cswap(aux, num_latent + i, num_latent + num_trash + i)
        qr.h(aux)
        qr.measure(aux, 0)

        def identity(x: torch.Tensor) -> torch.Tensor:
            return x

        return SamplerQNN(
            circuit=qr,
            input_params=[],
            weight_params=qr.parameters,
            interpret=identity,
            output_shape=2,
            sampler=Sampler(),
        )

__all__ = ["FastHybridEstimator"]
