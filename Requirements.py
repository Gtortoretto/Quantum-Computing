# General Imports

import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)

import pickle
import time as time_module
import random
import os
import inspect
import pathlib
import math

#Essentials

import IPython
import numpy as np
from bs4 import BeautifulSoup
import requests
import func_timeout
import pandas as pd
from itables import show
from scipy.optimize import curve_fit
from scipy import optimize

#from objproxies import *

import matplotlib.pyplot as plt

#Qiskit

import qiskit
from qiskit.circuit import Gate
from qiskit.circuit.library import *
from qiskit import transpile
from qiskit.providers.models import *
from qiskit_ibm_runtime.fake_provider import *
from qiskit.providers.fake_provider import *
from qiskit.quantum_info import Statevector
from qiskit import QuantumCircuit
from qiskit_aer import StatevectorSimulator
import qiskit_aer
from qiskit_nature.second_q.transformers import ActiveSpaceTransformer
from qiskit_algorithms import AdaptVQE
from qiskit.quantum_info import SparsePauliOp, Pauli

## Qiskit Nature
    
from qiskit_nature.units import *   #ANGSTROM

from qiskit_nature.second_q.drivers import *    #PySCFDriver

from qiskit_nature.second_q.mappers import * #JordanWignerMapper

from qiskit_nature.second_q.algorithms import GroundStateEigensolver

from qiskit_nature.second_q.problems import ElectronicStructureProblem
from qiskit_nature.second_q.problems import EigenstateResult

from qiskit_nature.second_q.circuit.library import *  #Ansatz, HF


## Qiskit Algorithms
    
from qiskit_algorithms.minimum_eigensolvers import VQE as  VQE_algorithms   #VQE
 
from qiskit_algorithms.optimizers import *    #SLSQP

from qiskit.circuit.library import EfficientSU2   #EfficientSU2

## Qiskit Estimators

from qiskit.primitives import Estimator as Estimator_Nature # Estimator Deprecating

from qiskit_aer.primitives.estimator import Estimator as Estimator_Aer

from qiskit_aer.primitives.estimator import EstimatorV2 as Estimator_AerV2


from qiskit_ibm_runtime import Estimator 

from qiskit_ibm_runtime import EstimatorV2

## Qiskit Noise Models

from qiskit_aer.noise import NoiseModel

from qiskit.providers.fake_provider import *

##Qiskit Runtime IBM

from qiskit_ibm_runtime import QiskitRuntimeService, Session, Options, Batch

## Braket

from braket.tracking import Tracker
from qiskit_braket_provider import *
from braket.aws import AwsDevice
from braket.devices import Devices
from braket.aws import AwsDevice, AwsQuantumTask

## Mitiq 

from mitiq import zne

#%matplotlib inline

plt.style.use('dark_background')

plt.rcParams.update({
    'figure.facecolor':   '#282c34',  
    'axes.facecolor':     '#1e1e1e',    
    'grid.color':         '#444444',

    'figure.figsize': (10, 6),
    'figure.dpi': 150
})    

class Quantum_Estimator:


    def __init__(self, circuit, observable = None, driver = None, mapper = None):

        self.circuit = circuit
        self.results = {}
        
        if observable is not None:
            
            self.observable = observable
            self.problem_type = "Classic"
            
        elif all(a is not None for a in (driver, mapper)):
            
            self.observable = mapper.map(driver.second_q_ops()[0])
            self.problem_type = "Molecule"
        
        else: 
            
            raise ValueError("Observable missing or Driver and Mapper missing.")
        
        self._fix_circuit_qubits()
    
    def _fix_circuit_qubits(self):

        num_qubits_observable = self.observable.num_qubits
        num_qubits_circuit = self.circuit.num_qubits
        
        if num_qubits_observable != num_qubits_circuit:
            
            new_circuit = QuantumCircuit(num_qubits_observable)

            for instruction in self.circuit.data:
                qubit_indices = [self.circuit.qubits.index(q) for q in instruction.qargs]
                
                if all(i < num_qubits_observable for i in qubit_indices):
                    new_circuit.append(instruction.operation, qubit_indices)
            
            self.circuit = new_circuit
    
    def run_statevector(self):

        state = Statevector.from_instruction(self.circuit)
        
        expectation_value = state.expectation_value(self.observable).real
        
        self.results['Exact'] = expectation_value 
        
        return expectation_value
            
        
        















































