import numpy as np
import math
import gudhi
from scipy.spatial.distance import directed_hausdorff
import matplotlib.pyplot as plt
import operator

import sys
import csv

class Dist2Bifilt:

    def __init__(self):
        self.bifilt_file = 'bifilt.npy'
        

    def load_bifilt(self, filename=None):
        self.bifilt = np.load(filename)
        if filename == None:
            self.bifilt = np.load(self.bifilt_file) 
        else:
            self.bifilt = np.load(filename) 
        self.bifilt = np.ceil(self.bifilt)
        #self.bifilt = self.bifilt[:,:10,:10]
        self.simplex_tree = [0 for i in range(self.bifilt.shape[0])]

    def compute_sim(self, verbose=False):
        for i in range(self.bifilt.shape[0]):
            rips_complex = gudhi.RipsComplex(distance_matrix=self.bifilt[i,:,:])
            self.simplex_tree[i] = rips_complex.create_simplex_tree(max_dimension=1)

        if verbose == True:
            result_str = 'Rips complex is of dimension ' + repr(self.simplex_tree[0].dimension()) + ' - ' + \
                repr(self.simplex_tree[0].num_simplices()) + ' simplices - ' + \
                repr(self.simplex_tree[0].num_vertices()) + ' vertices.'
            print(result_str)
            fmt = '%s ; %.2f'
            for filtered_value in self.simplex_tree[0].get_filtration():
                print(fmt % tuple(filtered_value))
            print('\n')

    def compute_bifilt(self, filename):
        simplex_list = []
        with open(filename, "a") as myfile:
            N = self.bifilt.shape[0]
            row = "--datatype bifiltration\n--xlabel time\n--ylabel distance\n \n# data starts here\n"
            myfile.write(row)
            for j in range(1,4):
                for filtered_value in self.simplex_tree[0].get_filtration():
                    if len(filtered_value[0]) != j:
                        continue
                    filt = []
                    for i in range(min(N,N)):
                        filt.append(i)
                        filt.append(self.simplex_tree[N-1-i].filtration(filtered_value[0]))
                    row = ''
                    for i in range(len(filtered_value[0])):
                        row = row + str(filtered_value[0][i]) + ' '
                    row = row + '; '
                    for i in range(len(filt)):
                        row = row + str(filt[i]) + ' '
                    row = row + '\n'
                    print(row)
                    myfile.write(row)
                
if __name__=="__main__":
    d2b = Dist2Bifilt()
    d2b.load_bifilt(sys.argv[1])
    d2b.compute_sim(verbose=False)
    if len(sys.argv) == 2:
        d2b.compute_bifilt('bifilt_red.txt')
    else:
        d2b.compute_bifilt(sys.argv[2])
