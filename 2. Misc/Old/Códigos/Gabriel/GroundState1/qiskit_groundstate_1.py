# %%
import io
import sys

import numpy as np 

import matplotlib.pyplot as plt

import pyscf as scf	

import qiskit as qk
from mpl_toolkits import mplot3d
from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_algorithms.minimum_eigensolvers import NumPyMinimumEigensolver
from qiskit_nature.second_q.algorithms import GroundStateEigensolver
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q import *
from qiskit.algorithms.minimum_eigensolvers import VQE
from qiskit_algorithms.optimizers import SLSQP
from qiskit.primitives import Estimator
from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD
from qiskit_aer.primitives import Estimator as AerEstimator
from qiskit_aer.noise import NoiseModel
from qiskit.providers.fake_provider import FakeVigo

# %% [markdown]
# # H<sub>2</sub> - Clássico - "sto3g" : Polynomial Fit

# %%
def groundstate_classico(a, base = "sto3g") :

    driver = PySCFDriver(
        atom= f"H 0 0 0; H 0 0 {a}",
        basis=base,
        charge=0,
        spin=0,
        unit=DistanceUnit.ANGSTROM,
    )

    solver = GroundStateEigensolver(
        JordanWignerMapper(),
        NumPyMinimumEigensolver(),
    )

    problem = driver.run()
    result = solver.solve(problem)

    return result.groundenergy + result.nuclear_repulsion_energy

# %%

intervalo  = np.concatenate(((intervalo_importante := np.linspace(0.1, (end := 1.5), 15)), np.linspace(end, 3, 10)))

coeficientes = np.polyfit(intervalo, (gs_energy := list(groundstate_classico(f'{a}') for a in intervalo)), 12)

coeficientes_importante = np.polyfit(intervalo_importante, (gs_energy_importante := gs_energy[:len(intervalo_importante)]), 12)


# %%

fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, figsize=(10, 7), gridspec_kw={'height_ratios': [3, 2], 'hspace': 0})

ax1.plot((x := np.linspace(min(intervalo), max(intervalo), 500)), (y := np.polyval(coeficientes, x)), label=f'NumPyPolyfit\nE = {(min_y := min(y)):.4f}')
ax1.plot((x_importante := np.linspace(min(intervalo_importante), max(intervalo_importante), 500)), (y_importante := np.polyval(coeficientes_importante, x_importante)), label=f'NumPyPolyfit - Importante\nE = {(min_y_importante := min(y_importante)):.4f}')
ax1.scatter(intervalo, gs_energy, c = 'b', label=f'NumPyMinimumEigensolver\nE = {(min_gs := min(gs_energy)):.4f}')

ax1.axvline(x=x[np.argmin(y)], linestyle='--', color='tab:blue')
ax1.axvline(x=x_importante[np.argmin(y_importante)], linestyle='--', color='tab:orange')
ax1.axvline(x=intervalo[np.argmin(gs_energy)], linestyle='--', color='b')

ax1.legend()

ax2.plot(intervalo, gs_energy - np.polyval(coeficientes, intervalo), 'o', label='Erro')
ax2.plot(intervalo_importante, gs_energy_importante - np.polyval(coeficientes_importante, intervalo_importante), 'o', label='Erro - Importante')
ax2.axhline(y=0, color='k', linestyle='--')

ax2.legend()

fig.suptitle('GroundState Energy H2')

ax2.set_xlabel('Distância (Å)')
ax1.set_ylabel('E $(E_f)$')
ax2.set_ylabel('Erro')


# %% [markdown]
# # H<sub>2</sub> - Clássico - "sto3g" : Comparando Resultados Clássicos

# %%


def NPMinimumEigensolver(a, base = "sto3g") :

    driver = PySCFDriver(
        atom= f"H 0 0 0; H 0 0 {a}",
        basis=base,
        charge=0,
        spin=0,
        unit=DistanceUnit.ANGSTROM,
    )

    solver = GroundStateEigensolver(
        JordanWignerMapper(),
        NumPyMinimumEigensolver(),
    )

    problem = driver.run()
    result = solver.solve(problem)

    return result.groundenergy + result.nuclear_repulsion_energy


def VariationalQuantumEigensolver(a, base = "sto3g"):
        
    driver = PySCFDriver(
        atom= f"H 0 0 0; H 0 0 {a}",
        basis=base,
        charge=0,
        spin=0,
        unit=DistanceUnit.ANGSTROM,
    )
    
    es_problem = driver.run()
    
    mapper = JordanWignerMapper()

    ansatz = UCCSD(
        es_problem.num_spatial_orbitals,
        es_problem.num_particles,
        mapper,
        reps=1,
        initial_state=HartreeFock(
            es_problem.num_spatial_orbitals,
            es_problem.num_particles,
            mapper,
        ),
    )
        
    vqe_solver = VQE(Estimator(), ansatz, SLSQP())
    vqe_solver.initial_point = [0.0] * ansatz.num_parameters
    
    calc = GroundStateEigensolver(mapper, vqe_solver)
    
    res = calc.solve(es_problem)

    return res.groundenergy + res.nuclear_repulsion_energy


def PySCF_HF(a, base = 'sto-3g'):

    mol = scf.M(
        atom = f'H 0 0 0; H 0 0 {a}',
        basis = base
    )

    mol_HF = mol.HF()
    
    sys.stdout = io.StringIO()
    
    a = mol_HF.kernel()
    
    sys.stdout = sys.__stdout__

    return a



def PySCF_KSDFT(a, base = 'sto-3g'):

    mol = scf.M(
        atom = f'H 0 0 0; H 0 0 {a}',
        basis = base
    )

    mol_HF = mol.KS()
    
    sys.stdout = io.StringIO()
    
    mol_HF.xc = 'b3lyp'
     
    a = mol_HF.kernel()
    
    sys.stdout = sys.__stdout__

    return a

 

# %%

intervalo  = np.concatenate(((intervalo_importante := np.linspace(0.1, (end := 1.5), 15)), np.linspace(end, 3, 10)))

sys.stdout = io.StringIO()

for a in (metodos := ['NPMinimumEigensolver', 'PySCF_HF', 'PySCF_KSDFT', 'VariationalQuantumEigensolver']) :
                
        exec(f"coeficientes_{a} = np.polyfit(intervalo, (gs_energy_{a} := list({a}(str(b)) for b in intervalo)), 12)")

        exec(f"coeficientes_importante_{a} = np.polyfit(intervalo_importante, (gs_energy_importante_{a} := gs_energy_{a}[:len(intervalo_importante)]), 12)")

sys.stdout = sys.__stdout__


# %%

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot()

groundstate_calculados = []

for a,b in zip(metodos, ['tab:blue','tab:orange','tab:purple', 'tab:cyan']):
    
    ax.plot((x_importante := np.linspace(min(intervalo_importante), max(intervalo_importante), 500)), (y_importante := np.polyval(eval(f"coeficientes_importante_{a}"), x_importante)), label=f'{a}\nE = {(min_y_importante := min(y_importante)):.4f}', color = b)
    ax.scatter(intervalo, eval(f"gs_energy_{a}"), color = b)

    ax.axvline(x=x_importante[np.argmin(y_importante)], linestyle='--', color=b)
    
    groundstate_calculados.append([a, min_y_importante])

ax.set_xlabel('Distância (Å)')
ax.set_ylabel('E $(E_f)$')

ax.set_title('GroundState Energy H2')

ax.legend()


# %%
groundstate_calculados

# %%
f"{groundstate_calculados[0][0]} - {groundstate_calculados[-1][0]} : {abs(groundstate_calculados[0][1] - groundstate_calculados[-1][1])}"

# %% [markdown]
# # H<sub>2</sub> - Clássico - "sto3g" : Comparando VQEs, Estimator vs AER (Noise)

# %%

def VQE_Estimator_Exact(a, base = "sto3g"):
        
    driver = PySCFDriver(
        atom= f"H 0 0 0; H 0 0 {a}",
        basis=base,
        charge=0,
        spin=0,
        unit=DistanceUnit.ANGSTROM,
    )
    
    es_problem = driver.run()
    
    mapper = JordanWignerMapper()

    ansatz = UCCSD(
        es_problem.num_spatial_orbitals,
        es_problem.num_particles,
        mapper,
        initial_state=HartreeFock(
            es_problem.num_spatial_orbitals,
            es_problem.num_particles,
            mapper,
        ),
    )

    vqe_solver = VQE(Estimator(), ansatz, SLSQP())
    vqe_solver.initial_point = [0.0] * ansatz.num_parameters
    
    calc = GroundStateEigensolver(mapper, vqe_solver)
    
    res = calc.solve(es_problem)

    return res.groundenergy + res.nuclear_repulsion_energy



def VQE_Estimator_Shots_Seeds(a, base = "sto3g", shots = 2048, seed = 42):
        
    driver = PySCFDriver(
        atom= f"H 0 0 0; H 0 0 {a}",
        basis=base,
        charge=0,
        spin=0,
        unit=DistanceUnit.ANGSTROM,
    )
    
    es_problem = driver.run()
    
    mapper = JordanWignerMapper()

    ansatz = UCCSD(
        es_problem.num_spatial_orbitals,
        es_problem.num_particles,
        mapper,
        initial_state=HartreeFock(
            es_problem.num_spatial_orbitals,
            es_problem.num_particles,
            mapper,
        ),
    )

    vqe_solver = VQE(Estimator(options={"shots": shots, "seed": seed}), ansatz, SLSQP())
    vqe_solver.initial_point = [0.0] * ansatz.num_parameters
    
    calc = GroundStateEigensolver(mapper, vqe_solver)
    
    res = calc.solve(es_problem)

    return res.groundenergy + res.nuclear_repulsion_energy



def VQE_AerEstimator_Exact_NoNoise(a, base = "sto3g"):
        
    driver = PySCFDriver(
        atom= f"H 0 0 0; H 0 0 {a}",
        basis=base,
        charge=0,
        spin=0,
        unit=DistanceUnit.ANGSTROM,
    )
    
    es_problem = driver.run()
    
    mapper = JordanWignerMapper()

    estimator_ = AerEstimator(
        run_options={"shots": None},
        approximation=True,
    )
    
    
    ansatz = UCCSD(
        es_problem.num_spatial_orbitals,
        es_problem.num_particles,
        mapper,
        initial_state=HartreeFock(
            es_problem.num_spatial_orbitals,
            es_problem.num_particles,
            mapper,
        ),
    )

    vqe_solver = VQE(estimator_, ansatz, SLSQP())
    vqe_solver.initial_point = [0.0] * ansatz.num_parameters
    
    calc = GroundStateEigensolver(mapper, vqe_solver)
    
    res = calc.solve(es_problem)
    
    return res.groundenergy + res.nuclear_repulsion_energy



def VQE_AerEstimator_Noise(a, base = "sto3g", shots = 2048*10, seed = 42, method = "density_matrix"):
        
    driver = PySCFDriver(
        atom= f"H 0 0 0; H 0 0 {a}",
        basis=base,
        charge=0,
        spin=0,
        unit=DistanceUnit.ANGSTROM,
    )
    
    es_problem = driver.run()
    
    mapper = JordanWignerMapper()

    device = FakeVigo()

    coupling_map = device.configuration().coupling_map
    noise_model = NoiseModel.from_backend(device)

    estimator_ = AerEstimator(
        backend_options={
        "method": method,
        "coupling_map": coupling_map,
        "noise_model": noise_model,
        },
        run_options={"seed": seed, "shots": shots},
        transpile_options={"seed_transpiler": seed},
    )
    
    
    ansatz = UCCSD(
        es_problem.num_spatial_orbitals,
        es_problem.num_particles,
        mapper,
        initial_state=HartreeFock(
            es_problem.num_spatial_orbitals,
            es_problem.num_particles,
            mapper,
        ),
    )

    vqe_solver = VQE(estimator_, ansatz, SLSQP())
    vqe_solver.initial_point = [0.0] * ansatz.num_parameters
    
    calc = GroundStateEigensolver(mapper, vqe_solver)
    
    res = calc.solve(es_problem)
    
    return res.groundenergy + res.nuclear_repulsion_energy


# %%
display(VQE_Estimator_Exact(1))
display(VQE_Estimator_Shots_Seeds(1))
display(VQE_AerEstimator_Exact_NoNoise(1))
display(VQE_AerEstimator_Noise(1, shots = 2048*10, seed = 200))

# %%
display(VQE_Estimator_Shots_Seeds(1))
display(VQE_Estimator_Shots_Seeds(1, seed=1))
display(VQE_Estimator_Shots_Seeds(1, seed=201))

# %%
display(VQE_AerEstimator_Noise(1, shots = 2048*10, seed = 200))
display(VQE_AerEstimator_Noise(1, shots = 2048*10, seed = 200, method='statevector'))

# %%
intervalo  = np.concatenate(((intervalo_importante := np.linspace(0.1, (end := 1.5), 15)), np.linspace(end, 3, 10)))

sys.stdout = io.StringIO()

for a in (metodos := ['VQE_Estimator_Exact', 'VQE_Estimator_Shots_Seeds', 'VQE_AerEstimator_Exact_NoNoise', 'VQE_AerEstimator_Noise']) :
                
        exec(f"coeficientes_{a} = np.polyfit(intervalo, (gs_energy_{a} := list({a}(str(b)) for b in intervalo)), 12)")

        exec(f"coeficientes_importante_{a} = np.polyfit(intervalo_importante, (gs_energy_importante_{a} := gs_energy_{a}[:len(intervalo_importante)]), 12)")

sys.stdout = sys.__stdout__

# %%
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot()

groundstate_calculados = []

for a,b in zip(metodos, ['tab:blue','tab:orange','tab:purple', 'tab:cyan']):
    
    ax.plot((x_importante := np.linspace(min(intervalo_importante), max(intervalo_importante), 500)), (y_importante := np.polyval(eval(f"coeficientes_importante_{a}"), x_importante)), label=f'{a}\nE = {(min_y_importante := min(y_importante)):.4f}', color = b)
    ax.scatter(intervalo, eval(f"gs_energy_{a}"), color = b)

    ax.axvline(x=x_importante[np.argmin(y_importante)], linestyle='--', color=b)
    
    groundstate_calculados.append([a, min_y_importante])

ax.set_xlabel('Distância (Å)')
ax.set_ylabel('E $(E_f)$')

ax.set_title('GroundState Energy H2')

ax.legend()

# %%
intervalo_importante = np.linspace(0.5, 1, 15)

for a in (metodos := ['VQE_Estimator_Exact', 'VQE_Estimator_Shots_Seeds', 'VQE_AerEstimator_Exact_NoNoise']) :

        exec(f"coeficientes_importante_{a} = np.polyfit(intervalo_importante, (gs_energy_importante_{a} := list({a}(str(b)) for b in intervalo_importante)), 12)")


# %%
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot()

groundstate_calculados = []

for a,b in zip(metodos, ['tab:blue','tab:orange','tab:purple']):
    
    ax.plot((x_importante := np.linspace(min(intervalo_importante), max(intervalo_importante), 500)), (y_importante := np.polyval(eval(f"coeficientes_importante_{a}"), x_importante)), label=f'{a}\nE = {(min_y_importante := min(y_importante)):.4f}', color = b)
    ax.scatter(intervalo_importante, eval(f"gs_energy_importante_{a}"), color = b)

    ax.axvline(x=x_importante[np.argmin(y_importante)], linestyle='--', color=b)
    
    groundstate_calculados.append([a, min_y_importante])

ax.set_xlabel('Distância (Å)')
ax.set_ylabel('E $(E_f)$')

ax.set_title('GroundState Energy H2')

ax.legend()

# %%
intervalo  = np.concatenate(((intervalo_importante := np.linspace(0.1, (end := 1.5), 15)), np.linspace(end, 3, 10)))

for a in (metodos := range(1000, 10000, 1000)):
                
        exec(f"coeficientes_{a} = np.polyfit(intervalo, (gs_energy_{a} := list(VQE_AerEstimator_Noise(str(b), shots = {a}) for b in intervalo)), 12)")

        exec(f"coeficientes_importante_{a} = np.polyfit(intervalo_importante, (gs_energy_importante_{a} := gs_energy_{a}[:len(intervalo_importante)]), 12)")

# %%
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot()

groundstate_calculados = []

for a,b in zip(metodos, ['tab:blue', 'tab:orange', 'tab:purple', 'tab:red', 'tab:green', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive']):
    
    ax.plot((x_importante := np.linspace(min(intervalo_importante), max(intervalo_importante), 500)), (y_importante := np.polyval(eval(f"coeficientes_importante_{a}"), x_importante)), label=f'{a}\nE = {(min_y_importante := min(y_importante)):.4f}', color = b)
    ax.scatter(intervalo, eval(f"gs_energy_{a}"), color = b)

    ax.axvline(x=x_importante[np.argmin(y_importante)], linestyle='--', color=b)
    
    groundstate_calculados.append([a, min_y_importante])

ax.set_xlabel('Distância (Å)')
ax.set_ylabel('E $(E_f)$')

ax.set_title('GroundState Energy H2')

ax.legend()

# %%
intervalo  = np.concatenate(((intervalo_importante := np.linspace(0.1, (end := 1.5), 15)), np.linspace(end, 3, 10)))

metodos = ['statevector', 'density_matrix']
                
coeficientes_statevector = np.polyfit(intervalo, (gs_energy_statevector := list(VQE_AerEstimator_Noise(str(b), method = 'statevector') for b in intervalo)), 12)

coeficientes_importante_statevector = np.polyfit(intervalo_importante, (gs_energy_importante_statevector := gs_energy_statevector[:len(intervalo_importante)]), 12)
                
coeficientes_density_matrix = np.polyfit(intervalo, (gs_energy_density_matrix := list(VQE_AerEstimator_Noise(str(b), method = 'density_matrix') for b in intervalo)), 12)

coeficientes_importante_density_matrix = np.polyfit(intervalo_importante, (gs_energy_importante_density_matrix := gs_energy_density_matrix[:len(intervalo_importante)]), 12)

# %%
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot()

groundstate_calculados = []

for a,b in zip(metodos, ['tab:blue', 'tab:orange']):
    
    ax.plot((x_importante := np.linspace(min(intervalo_importante), max(intervalo_importante), 500)), (y_importante := np.polyval(eval(f"coeficientes_importante_{a}"), x_importante)), label=f'{a}\nE = {(min_y_importante := min(y_importante)):.4f}', color = b)
    ax.scatter(intervalo, eval(f"gs_energy_{a}"), color = b)

    ax.axvline(x=x_importante[np.argmin(y_importante)], linestyle='--', color=b)
    
    groundstate_calculados.append([a, min_y_importante])

ax.set_xlabel('Distância (Å)')
ax.set_ylabel('E $(E_f)$')

ax.set_title('GroundState Energy H2')

ax.legend()


