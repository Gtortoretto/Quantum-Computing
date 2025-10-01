# General Imports

import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)

import time
import random
import pickle
import os
import IPython
import pprint
from tqdm.notebook import tqdm
from math import *
from tabulate import tabulate
from itables import init_notebook_mode

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

# Qiskit

from qiskit import QuantumCircuit, schedule
from qiskit.circuit import *
from qiskit.primitives import BackendEstimator
from qiskit.primitives import StatevectorEstimator, StatevectorSampler
from qiskit.providers.basic_provider import BasicSimulator
from qiskit import transpile
from qiskit.visualization import plot_histogram, plot_distribution
from qiskit.circuit.random import random_circuit
from qiskit.circuit.library import *
from qiskit.quantum_info import Operator, Statevector, DensityMatrix, SparsePauliOp, random_statevector
from qiskit.converters import *

from qiskit.providers.fake_provider import *
from qiskit_aer.noise import *

from qiskit_aer import *
from qiskit_aer.primitives import Estimator as Aer_EstimatorV1, EstimatorV2 as Aer_EstimatorV2
from qiskit_ibm_runtime import SamplerV2 as runtime_SamplerV2, EstimatorV2 as runtime_EstimatorV2, QiskitRuntimeService


plt.style.use('dark_background')

#init_notebook_mode(all_interactive=True)

plt.rcParams.update({
    'figure.facecolor':   '#282c34',  
    'axes.facecolor':     '#1e1e1e',    
    'grid.color':         '#444444',

    'figure.figsize': (10, 6),
    'figure.dpi': 150
})    

def get_current_directory():
    try:
        
        directory = os.path.dirname(os.path.abspath(__file__))

    except:
        
        ip = IPython.get_ipython()
        directory = None
        if '__vsc_ipynb_file__' in ip.user_ns:
            directory = os.path.dirname(ip.user_ns['__vsc_ipynb_file__'])
        
    return directory 

def salvar(a):
    
    script_dir = get_current_directory()
    
    dados_dir = os.path.join(script_dir, 'dados')
    
    os.makedirs(dados_dir, exist_ok=True)
    
    file_path = os.path.join(dados_dir, f'{a}.pickle')
    
    with open(file_path, 'wb') as f:
        pickle.dump(eval(a), f)
        
        
def abrir(a):
    
    script_dir = get_current_directory()
    
    file_path = os.path.join(script_dir, 'dados', f'{a}.pickle')
    
    with open(file_path, 'rb') as f:
        return pickle.load(f)

    
def salvar_obj(obj, filename):
    script_dir = get_current_directory()
    dados_dir = os.path.join(script_dir, 'dados')
    os.makedirs(dados_dir, exist_ok=True)
    
    file_path = os.path.join(dados_dir, f'{filename}.pickle')
    
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f)



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
            
            
            
    

            
        
        















































