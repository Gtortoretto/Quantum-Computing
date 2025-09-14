from braket.tracking import Tracker
from qiskit_braket_provider import BraketLocalBackend
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import BackendEstimator
from qiskit.circuit.library import TwoLocal
from qiskit_algorithms.optimizers import SLSQP
from qiskit_algorithms import VQE


t = Tracker().start()


local_simulator = BraketLocalBackend()

H2_op = SparsePauliOp.from_list([('II', -1.052373245772859), ('IZ', 0.39793742484318045), ('ZI', -0.39793742484318045), ('ZZ', -0.01128010425623538), ('XX', 0.18093119978423156)])

qi = BackendEstimator(local_simulator, options={'seed_simulator':42})
qi.set_transpile_options(seed_transpiler=42)

# Specify VQE configuration
ansatz = TwoLocal(rotation_blocks="ry", entanglement_blocks="cz")
slsqp = SLSQP(maxiter=1)
vqe = VQE(estimator=qi, ansatz=ansatz, optimizer=slsqp)

# Find the ground state
result = vqe.compute_minimum_eigenvalue(H2_op)
print(result)


print("Quantum Task Summary")
print(t.quantum_tasks_statistics())
print('Note: Charges shown are estimates based on your Amazon Braket simulator and quantum processing unit (QPU) task usage. Estimated charges shown may differ from your actual charges. Estimated charges do not factor in any discounts or credits, and you may experience additional charges based on your use of other services such as Amazon Elastic Compute Cloud (Amazon EC2).')
print(f"Estimated cost to run this example: {t.qpu_tasks_cost() + t.simulator_tasks_cost():.2f} USD")





