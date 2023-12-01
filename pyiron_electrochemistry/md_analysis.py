import matplotlib.pyplot as plt
import numpy as np
from pyiron_base import (
    GenericJob,
    GenericParameters,
    state,
    Executable,
    FlattenedStorage,)

class MDAnalysis(GenericJob):
    def __init__(self, project, job_name):
        super().__init__(project, job_name)
        self._init_project = project
        
    def add_jobs(self, job_list, different_project = None):
        if different_project is not None:
            self._project = different_project
        else:
            self._project = self._init_project   
        self._job_list = job_list
        
        self.energy = []
        self.positions = []
        
        for job_name in self._job_list:
            job = self._project.load(job_name, convert_to_object = False)
            # list append in the loop
            self.energy = np.concatenate((self.energy, job["output/generic/energy_tot"]))            
            if len(self.positions) == 0:
                self.positions = job["output/generic/positions"]
                self._structure = job["input/structure"].to_object()
            else:
                self.positions = np.concatenate((self.positions, job["output/generic/positions"]))
        # after the loop change it to numpy array
        
    def plot_energy_histogram(self, bins = 50):
        plt.hist(self.energy, bins = bins)
        plt.axvline(np.mean(self.energy), color ='Black')
        std = np.std(self.energy)
        plt.axvline(np.mean(self.energy)+std, color ='Black', linestyle = 'dashed')
        plt.axvline(np.mean(self.energy)-std, color ='Black', linestyle = 'dashed')
        plt.xlabel('energy (eV)')
        
    def get_rdf(self, el1, el2, rcut = 8, bins = 50, step_starting = 0, step_interval = 2):
        index1 = self._structure.select_index(el1)
        
        gofr = np.zeros(bins)
        Lbin = rcut / bins
        cell = self._structure.cell
        traj = self.positions[step_starting::step_interval]        
        G = np.array(cell) @ np.array(cell).T
        G_inv = np.linalg.inv(G)
        d_ij_list = []
        for index in index1:
            index2 = self._structure.select_index(el2)
            if index in index2:
                arg_where = np.argwhere( index2 == index)
                index2 = np.delete(index2, arg_where)
            diff = traj[:, [index], :] - traj[:, index2, :]
            # cell: 3x3
            diff_component = np.einsum('ilk,jk -> ilj',diff,np.array(cell))
            diff_component = np.einsum('ilj, kj-> ilk',diff_component, G_inv)
            diff_component =  np.mod(diff_component + 0.5, 1) - 0.5
            diff_correct = np.einsum('ilk, kj-> ilj',diff_component, np.array(cell))
            d_ij = np.linalg.norm(diff_correct, axis=2)   
            d_ij_list.append(d_ij.flatten())
        
        hist, bins_edge = np.histogram(d_ij_list, bins = bins)
        hist = hist.astype(float)
        # normalize
        V = cell[0,0] * cell[1,1] * cell[2,2]
        dr = bins_edge[1] - bins_edge[0]
        if el1 == el2:
            len_index2 = len(index2)-1
        else:
            len_index2 = len(index2)
        for ii in range(bins):
            r = bins_edge[ii] + 0.5* dr
            hist[ii] = V/(len(index1) * len_index2) * hist[ii]  / (4*np.pi*r**2*dr) / traj.shape[0]
        return hist, bins_edge

                
    def get_atomic_density(self, axis = 2, bins = 50):
        for el in set(self._structure.get_chemical_symbols()):
            atom_index = self._structure.select_index(el)
            plt.hist(self.positions[:,atom_index, axis].flatten(), alpha = 0.5, label = el, bins = bins)
        plt.legend()  
               
