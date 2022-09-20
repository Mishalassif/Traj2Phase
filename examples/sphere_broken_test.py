from traj2sim import *
import math

from mpl_toolkits import mplot3d
#%matplotlib widget
import numpy as np
import matplotlib.pyplot as plt

def plot_traj(list_traj):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    for i in range(len(list_traj)):
        xdata = list_traj[i][:,1]
        ydata = list_traj[i][:,2]
        zdata = list_traj[i][:,3]
        if i%2 == 0:
            ax.scatter3D(xdata, ydata, zdata, c='g')
        else:
            ax.scatter3D(xdata, ydata, zdata, c='r')

def sphere_point(theta, phi):
    return [math.sin(phi)*math.cos(theta), math.sin(phi)*math.sin(theta), -math.cos(phi)]

def great_circle_with_time(theta, length=10, noise=False, sigma=0.05, time_i=0, time_f=10):
    traj = np.zeros((time_f-time_i,4))
    for i in range(0, time_f-time_i):
        traj[i][1:] = sphere_point(theta, (time_i+i)*math.pi/length)
        traj[i][0] = sphere_point(theta, (time_i+i)*math.pi/length)[2]
        if noise == True:
            traj[i][1:] = np.add(traj[i],sigma*np.random.randn(1,3))
    return traj

t2s = Traj2Sim()

list_traj = []
N=50
length=40
for i in range(N):
    if i%2 == 0:
        list_traj.append(great_circle_with_time(2*i*math.pi/N, length, noise=False,time_i=int(0.0*length), time_f=int(0.8*length)))
    else:
        list_traj.append(great_circle_with_time(2*i*math.pi/N, length, noise=False,time_i=int(0.2*length), time_f=int(1.0*length)))

plot_traj(list_traj[:])
plt.show()
t2s.set_trajectories(list_traj[:3])
t2s.dist = 'custom'
t2s.compute_dist(verbose=False)
t2s.compute_sim(verbose=False)


print('Persistence intervals:')
#print(t2s.simplex_tree.persistence(homology_coeff_field=2, min_persistence=0.01))
#print(t2s.simplex_tree.persistence(homology_coeff_field=2, min_persistence=-1.0))
#t2s.display_persistence()
