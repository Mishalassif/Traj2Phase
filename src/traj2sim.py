import numpy as np
import gudhi
from scipy.spatial.distance import directed_hausdorff

from dtaidistance import dtw
from dtaidistance import dtw_ndim
from dtaidistance import dtw_visualisation as dtwvis

class Traj2Sim:

    def __init__(self):
        
        self.trajectories = []
        self.dist_mat = np.empty((1,1))

    def set_trajectories(self, list_traj):
        self.trajectories = list_traj
        self.dist_mat = np.empty((len(list_traj), len(list_traj)))

    def add_trajectories(self, list_traj):
        self.trajectories = self.trajectories + list_traj

    def norm(self, traj_1, traj_2):
        return np.linalg.norm(traj_1-traj_2)

    def compute_dist(self):
        for i in range(len(self.trajectories)):
            for j in range(len(self.trajectories)):
                #self.dist_mat[i,j] = directed_hausdorff(self.trajectories[i], self.trajectories[j])[0]
                self.dist_mat[i,j] = self.matching_dist(self.trajectories[i], self.trajectories[j])

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

    def compute_persistence(self):
        self.simplex_tree.compute_persistence()
        print(self.simplex_tree.persistence(homology_coeff_field=2, min_persistence=-1.0))
        print(self.simplex_tree.persistence_intervals_in_dimension(1))

    def _dist_to_line(self,l1, l2, p):
        t = max(min(-(np.dot(p-l1, l2-l1))/(np.linalg.norm(l2-l1)**2),1),0)
        return np.linalg.norm(l1 + t*(l2-l1) - p)

    def matching_dist(self,traj1, traj2):
        #s1 = np.array([0., 0, 1, 2, 1, 0, 1, 0, 0, 2, 1, 0, 0])
        #s2 = np.array([0., 1, 2, 3, 1, 0, 0, 0, 2, 1, 0, 0, 0])
        #path = dtw.warping_path(s1, s2)
        return dtw_ndim.distance(traj1, traj2)
'''
        len1 = traj1.shape[1]
        len2 = traj2.shape[1]
        dist_array = np.zeros((len1, len2))
        for j in range(len2):
            if j == 0:
                dist_array[len1-1,len2-1] = np.linalg.norm(traj1[:,len1-1]-traj2[:,len2-1])
                continue
            dist_array[len1-1,len2-1-j] = min(np.linalg.norm(traj1[:,len1-1]-traj2[:,len2-1-j]),
                    dist_array[len1-1, len2-1-j+1])

        temp = np.zeros((1,len2))
        for i in range(2,len1+1):
            for j in range(1,len2):
            
        return _dist_to_line(traj1[ind1], traj2[ind2])
'''

