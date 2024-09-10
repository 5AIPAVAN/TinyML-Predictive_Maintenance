# Import necessary libraries
from qiskit import QuantumCircuit, Aer, transpile, assemble
from qiskit.visualization import plot_histogram

# Step 1: Initialize a quantum circuit with 2 qubits and 2 classical bits
qc = QuantumCircuit(2, 2)

# Step 2: Apply quantum gates to the circuit
qc.h(0)  # Apply Hadamard gate to qubit 0
qc.cx(0, 1)  # Apply CNOT gate with qubit 0 as control and qubit 1 as target

# Step 3: Measure the qubits and store the result in classical bits
qc.measure([0, 1], [0, 1])

# Step 4: Choose a simulator backend
backend = Aer.get_backend('qasm_simulator')

# Step 5: Compile and execute the quantum circuit
compiled_circuit = transpile(qc, backend)
qobj = assemble(compiled_circuit)

# Step 6: Execute the circuit on the chosen backend
job = backend.run(qobj)

# Step 7: Get the result of the execution
result = job.result()

# Step 8: Visualize the result
counts = result.get_counts(qc)
print("Measurement result:", counts)

# Step 9: Plot the histogram of measurement outcomes
plot_histogram(counts)
