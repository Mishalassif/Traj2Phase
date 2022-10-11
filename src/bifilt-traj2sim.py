import numpy as np
import math
import gudhi
from scipy.spatial.distance import directed_hausdorff
import matplotlib.pyplot as plt
import operator

from matching_string_distance import MMSD

class Traj2BFSim:

    def __init__(self):
        
        self.trajectories = []
        self.MMSD = MMSD()

    def set_trajectories(self, list_traj):
        self.trajectories = list_traj
        self.dist_mat = []

    def add_trajectories(self, list_traj):
        self.trajectories = self.trajectories + list_traj

    def norm(self, traj_1, traj_2):
        return np.linalg.norm(traj_1-traj_2)

    def compute_dist(self, verbose=False):
        for i in range(len(self.trajectories)):
            for j in range(i, len(self.trajectories)):

                if verbose == True:
                    print('Custom distance between ' + str(i) + ', ' + str(j) + ': ' + str(self.dist_mat[i,j]))

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
        #self.simplex_tree.persistence(homology_coeff_field=2, min_persistence=-1.0))
        if self.simplex_tree.persistence_intervals_in_dimension(0).shape[0] != 0: 
            print('PH Dimension 0')
            gudhi.plot_persistence_diagram(self.simplex_tree.persistence_intervals_in_dimension(0))
            plt.show()
        if self.simplex_tree.persistence_intervals_in_dimension(1).shape[0] != 0: 
            print('PH Dimension 1')
            gudhi.plot_persistence_diagram(self.simplex_tree.persistence_intervals_in_dimension(1))
            plt.show()
        if self.simplex_tree.persistence_intervals_in_dimension(2).shape[0] != 0: 
            print('PH Dimension 2')
            gudhi.plot_persistence_diagram(self.simplex_tree.persistence_intervals_in_dimension(2))
            plt.show()
        if self.simplex_tree.persistence_intervals_in_dimension(2).shape[0] != 0: 
            print('PH Dimension 3')
            gudhi.plot_persistence_diagram(self.simplex_tree.persistence_intervals_in_dimension(3))
            plt.show()

