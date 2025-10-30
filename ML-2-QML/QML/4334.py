"""Quantum implementation of the hybrid Quantum‑NAT model.

The class mirrors the classical API but replaces the convolutional filter
and fully‑connected layer with parameterised quantum circuits:

* A 2‑layer CNN identical to the classical version.
* A 2‑qubit quanvolution circuit that encodes the pooled features via
  Ry rotations and a fixed random entangling layer.
* A 4‑qubit variational circuit with trainable Rx rotations.
* Measurement of Pauli‑Z expectation values yields a 4‑dimensional output.
* The output is batch‑normalised.

The circuit is executed on Qiskit’s Aer simulator.  The weight
parameters are exposed as `Parameter` objects and can be optimised
with Qiskit’s gradient‑based training utilities (e.g. `EstimatorQNN`)."""
