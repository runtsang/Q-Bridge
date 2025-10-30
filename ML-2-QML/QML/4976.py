import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf
import numpy as np
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector
from qiskit.primitives import Sampler as QiskitSampler
from typing import List, Sequence, Iterable, Any

class QuantumNATGen220(tq.QuantumModule):
    """
    Quantum implementation that mirrors the classical hybrid model.
    Encodes image features into a 4‑qubit device, processes them
    through a variational layer with attention‑style controlled‑rotation
    entanglement, and measures all qubits in the Pauli‑Z basis.
    A Qiskit sampler is exposed for expectation evaluation with optional
    finite‑shot noise.
    """
    class QLayer(tq.QuantumModule):
        def __init__(self):
            super().__init__()
            self.n_wires = 4
            self.random = tq.RandomLayer(n_ops=30, wires=list(range(self.n_wires)))
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)
            self.rz = tq.RZ(has_params=True, trainable=True)
            self.crx = tq.CRX(has_params=True, trainable=True)

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice):
            self.random(qdev)
            self.rx(qdev, wires=0)
            self.ry(qdev, wires=1)
            self.rz(qdev, wires=2)
            self.rx(qdev, wires=3)
            self.crx(qdev, wires=[0, 1])
            self.crx(qdev, wires=[1, 2])
            self.crx(qdev, wires=[2, 3])
            tqf.hadamard(qdev, wires=3)

    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.q_layer = self.QLayer()
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(self.n_wires)
        self._sampler = QiskitSampler()
        self._qiskit_circuit = None

    def _build_qiskit_circuit(self, params: Sequence[float]) -> ParameterVector:
        """
        Convert the internal variational parameters into a Qiskit circuit
        that can be passed to the Qiskit sampler for expectation evaluation.
        """
        if self._qiskit_circuit is None:
            param_vec = ParameterVector("θ", len(params))
            qc = QuantumCircuit(self.n_wires)
            for i, θ in enumerate(param_vec):
                qc.ry(θ, i)
            for i in range(self.n_wires - 1):
                qc.crx(param_vec[i], i, i + 1)
            self._qiskit_circuit = qc
        mapping = dict(zip(self._qiskit_circuit.parameters, params))
        return self._qiskit_circuit.assign_parameters(mapping, inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True)
        pooled = torch.nn.functional.avg_pool2d(x, 6).view(bsz, 16)
        self.encoder(qdev, pooled)
        self.q_layer(qdev)
        out = self.measure(qdev)
        return self.norm(out)

    def evaluate(
        self,
        observables: Iterable[Any],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[complex]]:
        """
        Evaluate expectation values of arbitrary qiskit operators for
        a list of parameter sets.  If `shots` is provided, the Qiskit
        sampler is used to emulate finite‑shot statistics; otherwise
        the expectation is computed exactly from the statevector
        returned by the internal circuit.
        """
        obs_list = list(observables)
        results: List[List[complex]] = []

        for params in parameter_sets:
            circuit = self._build_qiskit_circuit(params)
            if shots is None:
                state = Statevector.from_instruction(circuit)
                row = [state.expectation_value(obs) for obs in obs_list]
            else:
                job = self._sampler.run([circuit], shots=shots)
                counts = job[0].get_counts()
                sv = Statevector.from_counts(counts, dims=2**self.n_wires)
                row = [sv.expectation_value(obs) for obs in obs_list]
            results.append(row)

        if shots is None:
            return results

        rng = np.random.default_rng(seed)
        noisy = []
        for row in results:
            noisy_row = [rng.normal(val, max(1e-6, 1 / shots)) for val in row]
            noisy.append(noisy_row)
        return noisy

__all__ = ["QuantumNATGen220"]
