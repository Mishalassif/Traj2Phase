import numpy as np
import math
import gudhi
from scipy.spatial.distance import directed_hausdorff
import matplotlib.pyplot as plt
import operator



from dtaidistance import dtw
from dtaidistance import dtw_ndim
from dtaidistance import dtw_visualisation as dtwvis

class Traj2Sim:

    def __init__(self):
        
        self.trajectories = []
        self.dist_mat = np.empty((1,1))
        self.dist = 'custom'
        self.custom_pow=1

    def set_trajectories(self, list_traj):
        self.trajectories = list_traj
        self.dist_mat = np.empty((len(list_traj), len(list_traj)))

    def add_trajectories(self, list_traj):
        self.trajectories = self.trajectories + list_traj

    def norm(self, traj_1, traj_2):
        return np.linalg.norm(traj_1-traj_2)

    def compute_dist(self, verbose=False):
        for i in range(len(self.trajectories)):
            for j in range(len(self.trajectories)):
                if self.dist == 'custom':
                    self.dist_mat[i,j] = self.custom_broken_dist(self.trajectories[i], self.trajectories[j])
                    if verbose == True:
                        print('Custom distance between ' + str(i) + ', ' + str(j) + ': ' + str(self.dist_mat[i,j]))
                elif self.dist == 'hausdorff':
                    self.dist_mat[i,j] = directed_hausdorff(self.trajectories[i], self.trajectories[j])[0]
                elif self.dist == 'dtw':
                    self.dist_mat[i,j] = dtw_ndim.distance(self.trajectories[i], self.trajectories[j])

    def compute_sim(self, verbose=False):
        rips_complex = gudhi.RipsComplex(distance_matrix=self.dist_mat)
        self.simplex_tree = rips_complex.create_simplex_tree(max_dimension=4)
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

    def _dist_to_line(self,l1, l2, p):
        t = max(min(-(np.dot(p-l1, l2-l1))/(np.linalg.norm(l2-l1)**2),1),0)
        return np.linalg.norm(l1 + t*(l2-l1) - p)
    
    def _check_overlap(self, t1_i, t1_f, t2_i, t2_f):
        if t1_f < t2_i:
            return False, 0, 1, 0, 2
        elif t2_f < t1_i:
            return False, 0, 2, 0, 1

        if t1_i >= t2_i:
            ot_i = t1_i
            tr_i = 2
        else:
            ot_i = t2_i
            tr_i = 1

        if t1_f <= t2_f:
            ot_f = t1_f
            tr_f = 2
        else:
            ot_f = t2_f
            tr_f = 1
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
        
        d_i = max(d_i, 10e-8)
        d_f = max(d_f, 10e-8)
        return (ot_f-ot_i)*(1/math.pow(d_i, self.custom_pow) + 1/math.pow(d_f, self.custom_pow))/2.0

    def custom_broken_dist(self, traj1, traj2):
        dist = 0
        i = 0
        j = 0
        while(1):
            if i == len(traj1)-1 and j == len(traj2)-1:
                break
            #if(traj1[i+1,0]-traj1[i,0] < 0 or traj2[i+1,0]-traj2[i,0]<0):
            #    print('Time ordering Error!!!!')
            overlap, ot_i, tr_i, ot_f, tr_f = self._check_overlap(traj1[i,0], traj1[i+1,0], traj2[j,0], traj2[j+1,0])
            if overlap == True:
                #print('overlap at ' + str(i) + ', ' + str(j))
                dist += self._dist_integ(ot_i, ot_f, traj1[i], traj1[i+1], traj2[j], traj2[j+1])
                #print(dist)
                #print(ot_i)
                #print(ot_f)
                #print(tr_i)
                #print(tr_f)
                #print('%d %d %d %d', traj1[i,0], traj1[i+1,0], traj2[j,0], traj2[j+1,0])
            #else:
                #print('No overlap at ' + str(i) + ', ' + str(j))
            if tr_i == 1:
                if i == len(traj1)-2:
                    break
                i = i+1
                continue
            else:
                if j == len(traj2)-2:
                    break
                j = j+1
                continue
        if dist == 0:
            return np.inf
        return 1/math.pow(dist, 1/self.custom_pow)

    def longest_matching_time(self, traj1, traj2, epsilon):
        matching_times = [-1.0]
        i = 0
        j = 0
        curr_time = 0
        overlap_ongoing = False
        print(len(traj1))
        print(len(traj2))
        while(1):
            print('i,j: '+ str(i) + ',' + str(j))
            print('intervals: ' + '['+str(traj1[i,0]) +','+str(traj1[i+1,0])+'], ' +'['+str(traj2[j,0])+','+str(traj2[j+1,0])+']')

            if i >= len(traj1)-1 and j >= len(traj2)-1:
                print('Appending ' + str(curr_time))
                matching_times.append(curr_time)
                break
            if(traj1[i+1,0]-traj1[i,0] < 0 or traj2[i+1,0]-traj2[i,0]<0):
                print('Time ordering Error!!!!')
            overlap, ot_i, tr_i, ot_f, tr_f = self._check_overlap(traj1[i,0], traj1[i+1,0], traj2[j,0], traj2[j+1,0])
            t1_i = traj1[i]
            t2_i = traj2[j]
            t1_f = traj1[i+1]
            t2_f = traj2[j+1]
            if overlap == True:
                overlap_ongoing = True
                d_i = 10000
                d_f = 10000
                #print('Times ot_i, t1_i[0], t2_i[0], ot_f, t1_f[0], t2_f[0]')
                #print(str(ot_i) + ','+str(t1_i[0]) + ',' + str(t2_i[0]) + ',' + str(ot_f) + ',' + str(t1_f[0]) + ',' + str(t2_f[0]))
                if ot_i == t1_i[0]:
                    t2_ot_i = t2_i[1:] + ((ot_i-t2_i[0])/(t2_f[0]-t2_i[0]))*(t2_f[1:]-t2_i[1:])
                    d_i = np.linalg.norm(t2_ot_i - t1_i[1:])
                    #print('d_i for i, j and ot_i==1: ' + str(i) +',' + str(j)+','+str(d_i))
                elif ot_i == t2_i[0]:
                    t1_ot_i = t1_i[1:] + ((ot_i-t1_i[0])/(t1_f[0]-t1_i[0]))*(t1_f[1:]-t1_i[1:])
                    d_i = np.linalg.norm(t2_i[1:] - t1_ot_i)
                    #print('d_i for i, j and ot_i==2: ' + str(i) +',' + str(j)+','+str(d_i))

                if ot_f == t1_f[0]:
                    t2_ot_f = t2_i[1:] + ((ot_f-t2_i[0])/(t2_f[0]-t2_i[0]))*(t2_f[1:]-t2_i[1:])
                    d_f = np.linalg.norm(t2_ot_f - t1_f[1:])
                    #print('d_f for i, j and ot_f==1: ' + str(i) +',' + str(j)+','+str(d_f))
                elif ot_f == t2_f[0]:
                    t1_ot_f = t1_i[1:] + ((ot_f-t1_i[0])/(t1_f[0]-t1_i[0]))*(t1_f[1:]-t1_i[1:])
                    d_f = np.linalg.norm(t2_f[1:] - t1_ot_f)
                    #print('d_f for i, j and ot_f==2: ' + str(i) +',' + str(j)+','+str(d_f))

                if d_i <= epsilon:
                    if d_f <= epsilon:
                        curr_time += ot_f-ot_i
                    else:
                        curr_time += (ot_f-ot_i)*(epsilon-d_i)/(d_f-d_i)
                        print('Appending ' + str(curr_time))
                        matching_times.append(curr_time)
                        curr_time = 0
                elif d_i > epsilon:
                    if d_f <= epsilon:
                        curr_time = (ot_f-ot_i)*(epsilon-d_f)/(d_i-d_f)
                #print('Curr time: ' + str(curr_time))
            else:
                if overlap_ongoing == True:
                    #print('Appending ' + str(curr_time))
                    matching_times.append(curr_time)
                    curr_time = 0
                    overlap_ongoing = False

            if tr_i == 1:
                if i == len(traj1)-2:
                    #print('Appending ' + str(curr_time))
                    matching_times.append(curr_time)
                    break
                i = i+1
                continue
            else:
                if j == len(traj2)-2:
                    #print('Appending ' + str(curr_time))
                    matching_times.append(curr_time)
                    break
                j = j+1
                continue
        print(matching_times)
        delta = max(matching_times)
        if delta == -1.0:
            return np.inf
        else:
            return delta
    
    def smallest_matching_dist(self, traj1, traj2, delta):
        overlapping_dist = []
        i = 0
        j = 0
        curr_time = 0
        overlap_ongoing = False
        #print(len(traj1))
        #print(len(traj2))
        while(1):
            #print('i,j: '+ str(i) + ',' + str(j))
            #print('intervals: ' + '['+str(traj1[i,0]) +','+str(traj1[i+1,0])+'], ' +'['+str(traj2[j,0])+','+str(traj2[j+1,0])+']')

            if i >= len(traj1)-1 and j >= len(traj2)-1:
                #print('Appending ' + str(curr_time))
                matching_times.append(curr_time)
                break
            if(traj1[i+1,0]-traj1[i,0] < 0 or traj2[i+1,0]-traj2[i,0]<0):
                print('Time ordering Error!!!!')
            overlap, ot_i, tr_i, ot_f, tr_f = self._check_overlap(traj1[i,0], traj1[i+1,0], traj2[j,0], traj2[j+1,0])
            t1_i = traj1[i]
            t2_i = traj2[j]
            t1_f = traj1[i+1]
            t2_f = traj2[j+1]
            if overlap == True:
                overlap_ongoing = True
                
                #print('Times ot_i, t1_i[0], t2_i[0], ot_f, t1_f[0], t2_f[0]')
                #print(str(ot_i) + ','+str(t1_i[0]) + ',' + str(t2_i[0]) + ',' + str(ot_f) + ',' + str(t1_f[0]) + ',' + str(t2_f[0]))
                if ot_i == t1_i[0]:
                    t2_ot_i = t2_i[1:] + ((ot_i-t2_i[0])/(t2_f[0]-t2_i[0]))*(t2_f[1:]-t2_i[1:])
                    d_i = np.linalg.norm(t2_ot_i - t1_i[1:])
                    #print('d_i for i, j and ot_i==1: ' + str(i) +',' + str(j)+','+str(d_i))
                elif ot_i == t2_i[0]:
                    t1_ot_i = t1_i[1:] + ((ot_i-t1_i[0])/(t1_f[0]-t1_i[0]))*(t1_f[1:]-t1_i[1:])
                    d_i = np.linalg.norm(t2_i[1:] - t1_ot_i)
                    #print('d_i for i, j and ot_i==2: ' + str(i) +',' + str(j)+','+str(d_i))

                if ot_f == t1_f[0]:
                    t2_ot_f = t2_i[1:] + ((ot_f-t2_i[0])/(t2_f[0]-t2_i[0]))*(t2_f[1:]-t2_i[1:])
                    d_f = np.linalg.norm(t2_ot_f - t1_f[1:])
                    #print('d_f for i, j and ot_f==1: ' + str(i) +',' + str(j)+','+str(d_f))
                elif ot_f == t2_f[0]:
                    t1_ot_f = t1_i[1:] + ((ot_f-t1_i[0])/(t1_f[0]-t1_i[0]))*(t1_f[1:]-t1_i[1:])
                    d_f = np.linalg.norm(t2_f[1:] - t1_ot_f)
                    #print('d_f for i, j and ot_f==2: ' + str(i) +',' + str(j)+','+str(d_f))

                overlapping_dist.append([ot_i, d_i])

            else:
                if overlap_ongoing == True:
                    print('Overlapping times ended!')
                    break

            if tr_i == 1:
                if i == len(traj1)-2:
                    break
                i = i+1
                continue
            else:
                if j == len(traj2)-2:
                    break
                j = j+1
                continue
        #print('Overlapping distances: ' + str(overlapping_dist))
        
        if overlapping_dist[-1][0] - overlapping_dist[0][0] < delta:
            return np.inf

        i = 0
        j = len(overlapping_dist)-1
        eps_list = []
        #print('Length:')
        #print(len(overlapping_dist))
        while(1):
            if i == len(overlapping_dist):
                break
            if overlapping_dist[-1][0]-overlapping_dist[i][0] < delta:
                break
            if overlapping_dist[j][0]-overlapping_dist[i][0] >= delta:
                j = j-1
                continue
            else:
                #print('ij:')
                #print(i)
                #print(j)
                if i == j:
                    eps_list.append([i, overlapping_dist[i:j+1][0][1]])
                else:
                    eps_list.append([i, max(overlapping_dist[i:j+1], key=operator.itemgetter(1))[1]])
                    #print('right max:')
                    #print(max(overlapping_dist[i:j+1], key=operator.itemgetter(1))[1])
                i=i+1
                j = len(overlapping_dist)-1
        
        #print(eps_list)
        return min(eps_list, key=operator.itemgetter(1))
