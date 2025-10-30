import numpy as np
import torch
import strawberryfields as sf
from strawberryfields.ops import BSgate, Dgate, Kgate, Rgate, Sgate
import qiskit
from qiskit.circuit.random import random_circuit
from dataclasses import dataclass
from typing import Iterable, Sequence

@dataclass
class FraudLayerParameters:
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]

class FraudDetectionHybrid:
    def __init__(self, num_layers: int = 3, conv_kernel: int = 2, shots: int = 200) -> None:
        self.shots = shots
        self.qiskit_circuit, self.backend = self._build_qiskit_filter(conv_kernel)
        self.num_layers = num_layers
        self.program_builder = self._build_photonic_builder()

    def _build_qiskit_filter(self, kernel_size: int):
        n_qubits = kernel_size ** 2
        circuit = qiskit.QuantumCircuit(n_qubits)
        theta = [qiskit.circuit.Parameter(f'theta{i}') for i in range(n_qubits)]
        for i in range(n_qubits):
            circuit.rx(theta[i], i)
        circuit.barrier()
        circuit += random_circuit(n_qubits, 2)
        circuit.measure_all()
        backend = qiskit.Aer.get_backend('qasm_simulator')
        return circuit, backend

    def _build_photonic_builder(self):
        def builder(input_params, layers):
            prog = sf.Program(2)
            with prog.context as q:
                self._apply_layer(q, input_params, clip=False)
                for layer in layers:
                    self._apply_layer(q, layer, clip=True)
            return prog
        return builder

    def _apply_layer(self, q, params, clip):
        BSgate(params.bs_theta, params.bs_phi) | (q[0], q[1])
        for i, phase in enumerate(params.phases):
            Rgate(phase) | q[i]
        for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
            Sgate(r if not clip else self._clip(r, 5), phi) | q[i]
        BSgate(params.bs_theta, params.bs_phi) | (q[0], q[1])
        for i, phase in enumerate(params.phases):
            Rgate(phase) | q[i]
        for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
            Dgate(r if not clip else self._clip(r, 5), phi) | q[i]
        for i, k in enumerate(params.kerr):
            Kgate(k if not clip else self._clip(k, 1)) | q[i]

    def _clip(self, value: float, bound: float) -> float:
        return max(-bound, min(bound, value))

    def run(self, data: np.ndarray) -> float:
        data = np.reshape(data, (1, self.qiskit_circuit.num_qubits))
        param_binds = []
        for dat in data:
            bind = {}
            for i, val in enumerate(dat):
                bind[self.qiskit_circuit.parameters[i]] = np.pi if val > 0 else 0
            param_binds.append(bind)

        job = qiskit.execute(self.qiskit_circuit, self.backend,
                             shots=self.shots, parameter_binds=param_binds)
        result = job.result().get_counts(self.qiskit_circuit)
        prob = 0.0
        for key, val in result.items():
            ones = sum(int(bit) for bit in key)
            prob += ones * val
        prob /= self.shots * self.qiskit_circuit.num_qubits

        input_params = FraudLayerParameters(
            bs_theta=0.5, bs_phi=0.3,
            phases=(0.0, 0.0),
            squeeze_r=(0.2, 0.2),
            squeeze_phi=(0.0, 0.0),
            displacement_r=(prob, prob),
            displacement_phi=(0.0, 0.0),
            kerr=(0.0, 0.0)
        )
        layers = []
        for _ in range(self.num_layers - 1):
            layers.append(FraudLayerParameters(
                bs_theta=np.random.randn(),
                bs_phi=np.random.randn(),
                phases=(np.random.randn(), np.random.randn()),
                squeeze_r=(np.random.randn(), np.random.randn()),
                squeeze_phi=(np.random.randn(), np.random.randn()),
                displacement_r=(np.random.randn(), np.random.randn()),
                displacement_phi=(np.random.randn(), np.random.randn()),
                kerr=(np.random.randn(), np.random.randn())
            ))
        prog = self.program_builder(input_params, layers)
        eng = sf.Engine('fock', backend_options={"cutoff_dim": 5})
        result = eng.run(prog)
        n1 = result.state.expectation_value(sf.ops.Num(0))
        n2 = result.state.expectation_value(sf.ops.Num(1))
        return (n1 + n2).real / 2

__all__ = ["FraudDetectionHybrid"]
