import numpy as np
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit.library import ZFeatureMap
from qiskit.circuit import ParameterVector

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
    """
    Quantum fraud‑detection model.
    Implements a QCNN‑style circuit with convolution and pooling
    layers.  Each conv layer is parameterised by a FraudLayerParameters
    instance; pooling layers use fixed rotations.  The circuit is
    executed on an Aer qasm simulator and returns the probability of
    measuring |1> on the first qubit.
    """
    def __init__(self, input_params: FraudLayerParameters,
                 layers: Iterable[FraudLayerParameters]) -> None:
        self.backend = Aer.get_backend("qasm_simulator")
        self.shots = 1024
        self.circuit = self._build_circuit(input_params, layers)

    def _conv_layer(self, wires: Sequence[int], params: FraudLayerParameters) -> QuantumCircuit:
        qc = QuantumCircuit(*wires)
        qc.rx(params.bs_theta, wires[0])
        qc.ry(params.bs_phi, wires[1])
        qc.rz(params.phases[0], wires[0])
        qc.rz(params.phases[1], wires[1])
        return qc

    def _pool_layer(self, wires: Sequence[int]) -> QuantumCircuit:
        qc = QuantumCircuit(*wires)
        qc.cx(wires[0], wires[1])
        qc.rz(np.pi / 2, wires[1])
        return qc

    def _build_circuit(self, input_params: FraudLayerParameters,
                       layers: Iterable[FraudLayerParameters]) -> QuantumCircuit:
        num_qubits = 4
        qc = QuantumCircuit(num_qubits)
        # Classical feature map
        feature_map = ZFeatureMap(num_qubits)
        qc.append(feature_map, range(num_qubits))
        # QCNN layers
        for lay in layers:
            qc.append(self._conv_layer([0, 1], lay), [0, 1])
            qc.append(self._conv_layer([2, 3], lay), [2, 3])
            qc.append(self._pool_layer([0, 1]), [0, 1])
            qc.append(self._pool_layer([2, 3]), [2, 3])
        qc.measure_all()
        return qc

    def forward(self, data: Sequence[float]) -> float:
        """
        Execute the QCNN on the provided classical data.

        Parameters
        ----------
        data
            Iterable of length equal to the number of qubits (4).
        Returns
        -------
        float
            Probability of measuring |1> on the first qubit.
        """
        if len(data)!= self.circuit.num_qubits:
            raise ValueError("Data length must match number of qubits")
        # Bind feature map parameters
        param_binds = [{param: val for param, val in zip(self.circuit.parameters, data)}]
        job = execute(self.circuit, self.backend, shots=self.shots,
                      parameter_binds=param_binds)
        result = job.result()
        counts = result.get_counts(self.circuit)
        prob = sum(cnt for bitstr, cnt in counts.items() if bitstr[-1] == '1')
        return prob / self.shots

__all__ = ["FraudDetectionHybrid", "FraudLayerParameters"]
