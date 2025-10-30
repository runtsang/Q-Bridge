import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap, RealAmplitudes
from qiskit.primitives import Estimator
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN

def HybridQCNN() -> EstimatorQNN:
    """
    Quantum hybrid QCNN that interleaves convolution‑pooling blocks with a
    RealAmplitudes latent ansatz and a swap‑test style reconstruction check.
    """
    est = Estimator()

    # ---------- Convolution and pooling primitives ----------
    def conv_circuit(params: ParameterVector) -> QuantumCircuit:
        qc = QuantumCircuit(2)
        qc.rz(-np.pi/2, 1)
        qc.cx(1,0)
        qc.rz(params[0],0)
        qc.ry(params[1],1)
        qc.cx(0,1)
        qc.ry(params[2],1)
        qc.cx(1,0)
        qc.rz(np.pi/2,0)
        return qc

    def conv_layer(num_qubits: int, prefix: str) -> QuantumCircuit:
        qc = QuantumCircuit(num_qubits)
        params = ParameterVector(prefix, length=(num_qubits//2)*3)
        idx = 0
        for i in range(0, num_qubits, 2):
            qc.append(conv_circuit(params[idx:idx+3]), [i,i+1])
            qc.barrier()
            idx += 3
        return qc

    def pool_circuit(params: ParameterVector) -> QuantumCircuit:
        qc = QuantumCircuit(2)
        qc.rz(-np.pi/2,1)
        qc.cx(1,0)
        qc.rz(params[0],0)
        qc.ry(params[1],1)
        qc.cx(0,1)
        qc.ry(params[2],1)
        return qc

    def pool_layer(sources: list[int], sinks: list[int], prefix: str) -> QuantumCircuit:
        qc = QuantumCircuit(len(sources)+len(sinks))
        params = ParameterVector(prefix, length=len(sources)*3)
        idx = 0
        for s, t in zip(sources, sinks):
            qc.append(pool_circuit(params[idx:idx+3]), [s,t])
            qc.barrier()
            idx += 3
        return qc

    # ---------- Latent sub‑ansatz ----------
    def latent_ansatz(num_latent: int) -> QuantumCircuit:
        return RealAmplitudes(num_latent, reps=3)

    # ---------- Build full hybrid ansatz ----------
    num_qubits = 8  # example size
    qc = QuantumCircuit(num_qubits)
    fmap = ZFeatureMap(num_qubits)
    qc.append(fmap, range(num_qubits))

    qc.append(conv_layer(num_qubits, "c1"), range(num_qubits))
    qc.append(pool_layer(list(range(num_qubits//2)), list(range(num_qubits//2, num_qubits)), "p1"), range(num_qubits))

    qc.append(latent_ansatz(3), range(3))

    # Swap‑test style reconstruction (auxiliary qubit)
    qc.h(3)
    for i in range(3):
        qc.cswap(3, i, i+3)
    qc.h(3)

    meas_op = SparsePauliOp.from_list([("Z"+ "I"*(num_qubits-1), 1)])

    qnn = EstimatorQNN(
        circuit=qc.decompose(),
        observables=meas_op,
        input_params=fmap.parameters,
        weight_params=qc.parameters,
        estimator=est,
    )
    return qnn
