import numpy as np

class MSSD:

    def __init__(self):
        
        self.t1 = []
        self.t2 = []

        self.dist_mat = []

        self.sorted_dist = []
        self.sorted_dist_arg = []

        self.filt = []

        self.prev_substrings = []
        self.init_dict = dict()
        self.fin_dict = dict()

    def set_trajectories(self, t1, t2):
        
        if len(t1) > len(t2):
            self.t1 = t1
            self.t2 = t2
        else:
            self.t1 = t2
            self.t2 = t1

    def update_dist_mat(self):
        
        self.dist_mat = np.zeros((len(self.t1), len(self.t2)))
        for i in range(len(self.t1)):
            for j in range(len(self.t2)):
                self.dist_mat[i,j] = round(np.linalg.norm(self.t1[i][1:]
                        -self.t2[j][1:]), 3)

    def compute_matching_dist(self):
        self.update_dist_mat()
        M = len(self.t1)
        self.sorted_dist_arg = np.unravel_index(np.argsort(self.dist_mat, axis=None),
                self.dist_mat.shape)
        self.sorted_dist = self.dist_mat[self.sorted_dist_arg]
        
        self.init_dict[M*self.sorted_dist_arg[1][0]+self.sorted_dist_arg[0][0]] = M*self.sorted_dist_arg[1][0]+self.sorted_dist_arg[0][0]
        self.fin_dict[M*self.sorted_dist_arg[1][0]+self.sorted_dist_arg[0][0]] = M*self.sorted_dist_arg[1][0]+self.sorted_dist_arg[0][0]
        self.filt.append((self.sorted_dist[0], 1))

        for i in range(1, len(self.sorted_dist)):
            ind = (self.sorted_dist_arg[0][i], self.sorted_dist_arg[1][i])
            num_ind = M*ind[1] + ind[0]
            new_len = self._add_ind(num_ind)
            self.filt.append((self.sorted_dist[i], max(self.filt[-1][1], new_len)))

    def _add_ind(self, num_ind):

        M = len(self.t1)
        num_ind_next = num_ind+M+1
        num_ind_prev = num_ind-M-1

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
        
        return new_len

            
