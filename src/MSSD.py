import numpy as np

import time

class MSSD:
    def __init__(self):
        
        self.t1 = []
        self.t2 = []
        
        #self.string_struct = 'matrix'
        self.string_struct = 'dict'

        self.dist_mat = []
        self.sorted_dist = []
        self.sorted_dist_arg = []

        self.filt_eps = []
        self.init_dict = dict()
        self.fin_dict = dict()
        
        self.filt = []
    
        self._profiling = False

    def _reset(self):
        
        self.dist_mat = []
        self.sorted_dist = []
        self.sorted_dist_arg = []

        self.filt_eps = []
        self.init_dict = dict()
        self.fin_dict = dict()
        
        self.filt = []

    def set_trajectories(self, t1, t2):
        
        if len(t1) > len(t2):
            self.t1 = t1
            self.t2 = t2
        else:
            self.t1 = t2
            self.t2 = t1
    
        self._reset()
        if self.string_struct == 'matrix':
            self.init_dict = -np.ones((len(self.t1)*len(self.t2),))
            self.fin_dict = -np.ones((len(self.t1)*len(self.t2),))

    def update_dist_mat(self):
        
        self.dist_mat = np.zeros((len(self.t1), len(self.t2)))
        for i in range(len(self.t1)):
            for j in range(len(self.t2)):
                self.dist_mat[i,j] = round(np.linalg.norm(self.t1[i][1:]
                        -self.t2[j][1:]), 3)

    def compute_filt(self):
        self.update_dist_mat()
        M = len(self.t1)
        N = len(self.t2)
        
        if self._profiling == True:
            start = time.process_time()
        self.sorted_dist_arg = np.unravel_index(np.argsort(self.dist_mat, axis=None),
                self.dist_mat.shape)
        self.sorted_dist = self.dist_mat[self.sorted_dist_arg]
        if self._profiling == True:
            print('Time taken in Sorting stage: ' + str(time.process_time() - start))
            start = time.process_time()

        self.init_dict[M*self.sorted_dist_arg[1][0]+self.sorted_dist_arg[0][0]] = M*self.sorted_dist_arg[1][0]+self.sorted_dist_arg[0][0]
        self.fin_dict[M*self.sorted_dist_arg[1][0]+self.sorted_dist_arg[0][0]] = M*self.sorted_dist_arg[1][0]+self.sorted_dist_arg[0][0]
        self.filt_eps.append((self.sorted_dist[0], 1))
        
        for i in range(1, len(self.sorted_dist)):
            ind = (self.sorted_dist_arg[0][i], self.sorted_dist_arg[1][i])
            num_ind = M*ind[1] + ind[0]
            new_len = self._add_ind(num_ind)
            self.filt_eps.append((self.sorted_dist[i], max(self.filt_eps[-1][1], new_len)))

        if self._profiling == True:
            print('Time taken in Main iteration stage with ' + self.string_struct + ' data structure: ' + str(time.process_time() - start))
            start = time.process_time()

        t_curr = N
        self.filt.append([self.filt_eps[-1][0], self.filt_eps[-1][1]])
        for i in range(1, len(self.sorted_dist)):
            if t_curr == self.filt_eps[-1-i][1]:
                self.filt[-1][0] = self.filt_eps[-1-i][0]
            else:
                self.filt.append([self.filt_eps[-1-i][0], self.filt_eps[-1-i][1]])
                t_curr = self.filt[-1][1]
        self.filt.reverse()
        tmp = [0.0 for j in range(N)]
        for j in range(self.filt[0][1]):
            tmp[j] = [self.filt[0][0], j+1]
        for i in range(1, len(self.filt)):
            for j in range(self.filt[i-1][1], self.filt[i][1]):
                tmp[j] = [self.filt[i][0], j+1]
        self.filt = np.zeros((N,))
        for i in range(N):
            self.filt[i] = tmp[i][0]
        self.clear_memory()
        if self._profiling == True:
            print('Time taken in Post processing stage: ' + str(time.process_time() - start))

        return self.filt
    
    def metric(self, met='inf', t_thresh=5):
        if met == 'inf':
            return self.inf_metric()
        if met == 't_thresh':
            return self.t_thresh_metric(t_thresh)

    def inf_metric(self):
        return min([x[0]/x[1] for x in self.filt])

    def t_thresh_metric(self, t_thresh=5):
        return self.filt[t_thresh-1]
    
    def clear_memory(self):
        self.filt_eps.clear()
        if self.string_struct == 'dict':
            self.init_dict.clear()
            self.fin_dict.clear()
        elif self.string_struct == 'matrix':
            self.init_dict = np.empty((1,1))
            self.fin_dict = np.empty((1,1))

    def _add_ind(self, num_ind):
        M = len(self.t1)
        N = len(self.t2)
        num_ind_next = num_ind+M+1
        num_ind_prev = num_ind-M-1
        
        if self.string_struct == 'dict':
            if num_ind_next in self.init_dict and num_ind_prev in self.fin_dict:
                new_fin_ind = self.init_dict[num_ind_next]
                new_init_ind = self.fin_dict[num_ind_prev]
                self.init_dict.pop(num_ind_next)
                self.fin_dict.pop(num_ind_prev)
                self.init_dict[new_init_ind] = new_fin_ind
                self.fin_dict[new_fin_ind] = new_init_ind
                new_len = (new_fin_ind%M)-(new_init_ind%M)+1
            elif num_ind_next in self.init_dict and num_ind_prev not in self.fin_dict:
                new_fin_ind = self.init_dict[num_ind_next]
                new_init_ind = num_ind
                self.init_dict.pop(num_ind_next)
                self.init_dict[new_init_ind] = new_fin_ind
                self.fin_dict[new_fin_ind] = new_init_ind
                new_len = (new_fin_ind%M)-(new_init_ind%M)+1
            elif num_ind_next not in self.init_dict and num_ind_prev in self.fin_dict:
                new_fin_ind = num_ind
                new_init_ind = self.fin_dict[num_ind_prev]
                self.fin_dict.pop(num_ind_prev)
                self.fin_dict[new_fin_ind] = new_init_ind
                self.init_dict[new_init_ind] = new_fin_ind
                new_len = (new_fin_ind%M)-(new_init_ind%M)+1
            elif num_ind_next not in self.init_dict and num_ind_prev not in self.fin_dict:
                new_fin_ind = num_ind
                new_init_ind = num_ind
                self.fin_dict[new_fin_ind] = new_init_ind
                self.init_dict[new_init_ind] = new_fin_ind
                new_len = (new_fin_ind%M)-(new_init_ind%M)+1
        
        elif self.string_struct == 'matrix':
            if (num_ind_next < M*N and self.init_dict[num_ind_next] >= 0) and (num_ind_prev >= 0 and self.fin_dict[num_ind_prev] >= 0):
                new_fin_ind = int(self.init_dict[num_ind_next])
                new_init_ind = int(self.fin_dict[num_ind_prev])
                self.init_dict[num_ind_next] = -1
                self.fin_dict[num_ind_prev] = -1
                self.init_dict[new_init_ind] = new_fin_ind
                self.fin_dict[new_fin_ind] = new_init_ind
                new_len = (new_fin_ind%M)-(new_init_ind%M)+1
            elif (num_ind_next < M*N and self.init_dict[num_ind_next] >= 0) and (num_ind_prev < 0 or self.fin_dict[num_ind_prev] < 0):
                new_fin_ind = int(self.init_dict[num_ind_next])
                new_init_ind = num_ind
                self.init_dict[num_ind_next] = -1
                self.init_dict[new_init_ind] = new_fin_ind
                self.fin_dict[new_fin_ind] = new_init_ind
                new_len = (new_fin_ind%M)-(new_init_ind%M)+1
            elif (num_ind_next >= M*N or self.init_dict[num_ind_next] < 0) and (num_ind_prev >= 0 and self.fin_dict[num_ind_prev] >= 0):
                new_fin_ind = num_ind
                new_init_ind = int(self.fin_dict[num_ind_prev])
                self.fin_dict[num_ind_prev] = -1
                self.fin_dict[new_fin_ind] = new_init_ind
                self.init_dict[new_init_ind] = new_fin_ind
                new_len = (new_fin_ind%M)-(new_init_ind%M)+1
            elif (num_ind_next >= M*N or self.init_dict[num_ind_next] < 0) and (num_ind_prev < 0 or self.fin_dict[num_ind_prev] < 0):
                new_fin_ind = num_ind
                new_init_ind = num_ind
                self.fin_dict[new_fin_ind] = new_init_ind
                self.init_dict[new_init_ind] = new_fin_ind
                new_len = (new_fin_ind%M)-(new_init_ind%M)+1
        return new_len
