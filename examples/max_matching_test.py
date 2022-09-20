from traj2sim import *
import math

from mpl_toolkits import mplot3d
#%matplotlib widget
import numpy as np
import matplotlib.pyplot as plt

t2s = Traj2Sim()

length = 20
traj1 = np.zeros((length,4))
traj2 = np.zeros((length,4))

t_i1=0.0
t_i2=0.5
time1 = [t_i1 +(i/length)*5.0 for i in range(length)]
time2 = [t_i2 +(i/length)*5.0 for i in range(length)]

y_i1=2
y_i2=2.15
y1 = [y_i1 +(i/length)*3.0 for i in range(length)]
y2 = [y_i2 +(i/length)*3.0 for i in range(length)]

for i in range(int(length/3), length):
    y1[i] = y1[int(length/3)-1]-(i/length)*2.0

for i in range(int(length/3)+2, length):
    y2[i] = y2[int(length/3)+1]-(i/length)*4.0

for i in range(length):
    traj1[i][0] = time1[i]
    traj2[i][0] = time2[i]
    traj1[i][1] = time1[i]
    traj2[i][1] = time2[i]
    traj1[i][2] = y1[i]
    traj2[i][2] = y2[i]
    traj1[i][3] = time1[i]
    traj2[i][3] = time2[i]

eps=0.9

print(t2s.longest_matching_time(traj1, traj2, eps))

plt.plot(time1, y1)
plt.plot(time2, y2)
for i in range(length):
    print(np.sign(y2[i]-y1[i]))
    plt.plot([time1[i], time1[i]], [y1[i]+eps/2, y1[i]-eps/2],c='g')
    plt.plot([time2[i], time2[i]], [y2[i]-eps/2, y2[i]+eps/2],c='r')
plt.show()
