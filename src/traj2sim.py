import numpy as np
import math
import gudhi
from scipy.spatial.distance import directed_hausdorff
import matplotlib.pyplot as plt
import operator

from dtaidistance import dtw
from dtaidistance import dtw_ndim
from dtaidistance import dtw_visualisation as dtwvis

from MSSD import MSSD

from alive_progress import alive_bar
import time


class Traj2Sim:

    def __init__(self):
        
        self.trajectories = []
        self.dist_mat = np.empty((1,1))
        self.dist = 'integral'
        self.custom_pow=1
        self.MSSD = []

    def set_trajectories(self, list_traj):
        self.trajectories = list_traj
        self.dist_mat = np.empty((len(list_traj), len(list_traj)))
        self.MSSD = [[MSSD() for j in range(i+1,len(list_traj))] for i in range(len(list_traj))]

    def add_trajectories(self, list_traj):
        self.trajectories = self.trajectories + list_traj

    def norm(self, traj_1, traj_2):
        return np.linalg.norm(traj_1-traj_2)

    def compute_dist(self, verbose=False):
        for i in range(len(self.trajectories)):
            for j in range(len(self.trajectories)):
                if self.dist == 'hausdorff':
                    self.dist_mat[i,j] = directed_hausdorff(self.trajectories[i], self.trajectories[j])[0]
                elif self.dist == 'dtw':
                    self.dist_mat[i,j] = dtw_ndim.distance(self.trajectories[i], self.trajectories[j])
                if verbose == True:
                    print('Custom distance between ' + str(i) + ', ' + str(j) + ': ' + str(self.dist_mat[i,j]))
        
    def compute_mssd(self, verbose=False, met='t_thresh', thresh=5):
        with alive_bar(int(len(self.trajectories)*(len(self.trajectories)+1)/2), force_tty=True) as bar:
            for i in range(len(self.trajectories)):
                for j in range(i+1, len(self.trajectories)):
                    self.MSSD[i][j-i-1].set_trajectories(self.trajectories[i], self.trajectories[j])
                    self.dist_mat[i,j] = self.MSSD[i][j-i-1].compute_filt(met=met, t_thresh=thresh)
                    self.dist_mat[j,i] = self.dist_mat[i,j]
                    bar()
                    if verbose == True:
                        print('MSSD distance between ' + str(i) + ', ' + str(j) + ': ' + str(self.dist_mat[i,j]))
            for i in range(len(self.trajectories)):
                self.dist_mat[i,i] = 0
            return
    
    def update_mssd(self, verbose=False, met='t_thresh', thresh=5):
        with alive_bar(int(len(self.trajectories)*(len(self.trajectories)+1)/2), force_tty=True) as bar:
            for i in range(len(self.trajectories)):
                for j in range(i+1, len(self.trajectories)):
                    self.dist_mat[i,j] = self.MSSD[i][j-i-1].metric(met=met, t_thresh=thresh)
                    self.dist_mat[j,i] = self.dist_mat[i,j]
                    bar()
                    if verbose == True:
                        print('MSSD distance between ' + str(i) + ', ' + str(j) + ': ' + str(self.dist_mat[i,j]))
            for i in range(len(self.trajectories)):
                self.dist_mat[i,i] = 0
            return

    def compute_sim(self, verbose=False):
        rips_complex = gudhi.RipsComplex(distance_matrix=self.dist_mat)
        self.simplex_tree = rips_complex.create_simplex_tree(max_dimension=3)
        if verbose == True:
            result_str = 'Rips complex is of dimension ' + repr(self.simplex_tree.dimension()) + ' - ' + \
                repr(self.simplex_tree.num_simplices()) + ' simplices - ' + \
                repr(self.simplex_tree.num_vertices()) + ' vertices.'
            print(result_str)
            fmt = '%s -> %.2f'
            for filtered_value in self.simplex_tree.get_filtration():
                print(fmt % tuple(filtered_value))
            print('\n')

    def display_persistence(self):
        self.simplex_tree.compute_persistence(min_persistence=-0.1)
        for i in range(3):
            if self.simplex_tree.persistence_intervals_in_dimension(i).shape[0] != 0: 
                print('PH Dimension ' + str(i))
                gudhi.plot_persistence_diagram(self.simplex_tree.persistence_intervals_in_dimension(0))
                plt.show()
