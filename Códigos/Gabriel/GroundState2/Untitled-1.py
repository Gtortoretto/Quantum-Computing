# %%
#Python Default 

#Essentials

import numpy as np
from objproxies import *

#Qiskit
    
## Qiskit Nature
    
from qiskit_nature.units import *   #ANGSTROM

from qiskit_nature.second_q.drivers import *    #PySCFDriver

from qiskit_nature.second_q.mappers import * #JordanWignerMapper

from qiskit_nature.second_q.algorithms import GroundStateEigensolver

from qiskit_nature.second_q.problems import ElectronicStructureProblem

from qiskit_nature.second_q.circuit.library import *  #Ansatz, HF

## Qiskit Algorithms
    
from qiskit.algorithms.minimum_eigensolvers import *    #VQE
 
from qiskit.algorithms.optimizers import *    #SLSQP

## Qiskit Primitives

from qiskit.primitives import Estimator as Estimator_Nature

from qiskit_aer.primitives.estimator import Estimator as Estimator_Aer


# %% [markdown]
# # Definindo Funções

# %%
class Error_Qiskit:
    
    pass

class GroundState_H2:
    
    class Driver :
        
        def __init__(self, a, base, driver, charge, spin, unit):
            
            self._drivers = driver(
                atom= f"H 0 0 0; H 0 0 {a}",
                basis=base,
                charge=charge,
                spin=spin,
                unit=unit,
            )

        def run_driver(self) -> ElectronicStructureProblem:
            
            return self._drivers.run()
    
    def __init__(self, base = 'sto3g', unit = DistanceUnit.ANGSTROM, driver = PySCFDriver, charge = 0, spin = 0):
        
        self._problem = None
        self._dist = None
        self._base = base
        self._unit = unit
        self._driver = driver
        self._charge = charge
        self._spin = spin
        self._mapper = None
    
    def mapper(self):
        
        return self._mapper
    
    def problem (self, a = None) -> ElectronicStructureProblem:
        
        dist = self._dist if self._dist != None else a
        
        return self.Driver(dist, self._base, self._driver, self._charge, self._spin, self._unit).run_driver()
        
    def groundstate(self, a, mapper = JordanWignerMapper(), solver = NumPyMinimumEigensolver) -> float:
        
        self._mapper = mapper
        self._dist = a
        self._problem = self.problem(a)
    
        return (result := GroundStateEigensolver(mapper, solver()).solve(self._problem)).groundenergy + result.nuclear_repulsion_energy
    
    def groundstate_curve(self, a, mapper = JordanWignerMapper(), solver = NumPyMinimumEigensolver()) -> iter:
        
        self._mapper = mapper
        
        for dist in a:
    
            self._dist = a
            self._problem = self.problem(a)
    
            yield self.groundstate(dist, mapper, solver)
    
class Solver(GroundState_H2):
    
    def __init__(self, estimator = Estimator_Nature, ansatz = UCCSD, optimizer = SLSQP):
        
        super().__init__()
        self._estimator, self.estimator = estimator(), estimator
        self._optimizer, self.optimizer = optimizer(), optimizer
        self._ansatz, self.ansatz = ansatz(), ansatz
    
    def Ansatz(self):
    
        self._ansatz = self.ansatz(
            self._problem.num_spatial_orbitals,
            self._problem.num_particles,
            self._mapper,
            initial_state=HartreeFock(
            self._problem.num_spatial_orbitals,
            self._problem.num_particles,
            self._mapper)
            )
           
        
    def estimator(self, shots = 0, seed = np.random.randint(1, 1000), noise = None):
        
        if self.estimator == Estimator_Nature:
            
            self._estimator = self.estimator(options = {'shots': shots, 'seed': seed}) if shots != 0 else self.estimator()
        
        else:
            
            if shots == 0:
                
                self._estimator = self.estimator(run_options={"shots": None}, approximation=True)
        
            elif noise == None:
                
                self._estimator = self.estimator(run_options={"shots": shots, 'seed': seed}, transpile_options={"seed_transpiler": seed})
        
        return
    
    def vqe(self):
    
        return VQE(self._estimator, self._ansatz, self._optimizer)
    


GroundState_H2().groundstate(1, solver = lambda : Solver().Ansatz().vqe())
