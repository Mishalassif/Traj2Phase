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
        self.dist = 'custom'

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
                if self.dist == 'custom':
                    self.dist_mat[i,j] = self.custom_broken_dist(self.trajectories[i], self.trajectories[j])
                    print('Custom distance between ' + str(i) + ', ' + str(j) + ': ' + str(self.dist_mat[i,j]))
                elif self.dist == 'hausdorff':
                    self.dist_mat[i,j] = directed_hausdorff(self.trajectories[i], self.trajectories[j])[0]
                elif self.dist == 'dtw':
                    self.dist_mat[i,j] = dtw_ndim.distance(self.trajectories[i], self.trajectories[j])

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
        self.simplex_tree.compute_persistence()
        #self.simplex_tree.persistence(homology_coeff_field=2, min_persistence=-1.0))
        if self.simplex_tree.persistence_intervals_in_dimension(0).shape[0] != 0: 
            gudhi.plot_persistence_diagram(self.simplex_tree.persistence_intervals_in_dimension(0))
        if self.simplex_tree.persistence_intervals_in_dimension(1).shape[0] != 0: 
            gudhi.plot_persistence_diagram(self.simplex_tree.persistence_intervals_in_dimension(1))
        if self.simplex_tree.persistence_intervals_in_dimension(2).shape[0] != 0: 
            gudhi.plot_persistence_diagram(self.simplex_tree.persistence_intervals_in_dimension(2))

    def _dist_to_line(self,l1, l2, p):
        t = max(min(-(np.dot(p-l1, l2-l1))/(np.linalg.norm(l2-l1)**2),1),0)
        return np.linalg.norm(l1 + t*(l2-l1) - p)
    
    def _check_overlap(self, t1_i, t1_f, t2_i, t2_f):
        if t1_f < t2_i:
            return False, 0, 1, 0, 2
        elif t2_f < t1_i:
            return False, 0, 2, 0, 1

        if t1_i > t2_i:
            ot_i = t1_i
            tr_i = 1
        else:
            ot_i = t2_i
            tr_i = 2

        if t1_f < t2_f:
            ot_f = t1_f
            tr_f = 1
        else:
            ot_f = t2_f
            tr_f = 2
        return True, ot_i, tr_i, ot_f, tr_f

    def _dist_integ(self, ot_i, ot_f, t1_i, t1_f, t2_i, t2_f):
        if ot_i == t1_i[0]:
            t2_ot_i = t2_i[1:] + ((ot_i-t2_i[0])/(t2_f[0]-t2_i[0]))*(t2_f[1:]-t2_i[1:])
            d_i = np.linalg.norm(t2_ot_i - t1_i[1:])
        elif ot_i == t2_i[0]:
            t1_ot_i = t1_i[1:] + ((ot_i-t1_i[0])/(t1_f[0]-t1_i[0]))*(t1_f[1:]-t1_i[1:])
            d_i = np.linalg.norm(t2_i[1:] - t1_ot_i)

        if ot_f == t1_f[0]:
            t2_ot_f = t2_i[1:] + ((ot_f-t2_i[0])/(t2_f[0]-t2_i[0]))*(t2_f[1:]-t2_i[1:])
            d_f = np.linalg.norm(t2_ot_f - t1_f[1:])
        elif ot_f == t2_f[0]:
            t1_ot_f = t1_i[1:] + ((ot_f-t1_i[0])/(t1_f[0]-t1_i[0]))*(t1_f[1:]-t1_i[1:])
            d_f = np.linalg.norm(t2_f[1:] - t1_ot_f)

        return (ot_f-ot_i)*(1/d_i + 1/d_f)/2.0


    def custom_broken_dist(self, traj1, traj2):
        dist = 0
        i = 0
        j = 0
        while(1):
            if i == len(traj1)-1 or j == len(traj2)-1:
                break
            overlap, ot_i, tr_i, ot_f, tr_f = self._check_overlap(traj1[i,0], traj1[i+1,0], traj2[j,0], traj2[j+1,0])
            if overlap == True:
                dist += self._dist_integ(ot_i, ot_f, traj1[i], traj1[i+1], traj2[j], traj2[j+1])
            if ot_i == 1:
                i = i+1
                continue
            else:
                j = j+1
                continue
        return 1/dist
