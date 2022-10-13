# Traj2Phase

Recovering the phase space of a dynamical system from broken pieces of trajectories

## Package Requirements

Install the following packages

`pip install gudhi dtw-python scipy scikit-tda alive_progress`

## Setup

Add the src directory to PYTHONPATH using the command

`source setup.sh`

## Example results

Flow lines            |  Phase space homology
:-------------------------:|:-------------------------:
![Lorenz attractor](images/lorenz.png) |  ![Lorenz attractor](images/lorenz_pdgm.png)
![Torus](images/torus.png) |  ![Torus](images/torus_pdgm.png)
![Torus](images/torus_wind.png) |  ![Torus](images/torus_wind_pdgm.png)
![Torus](images/sphere.png) |  ![Torus](images/sphere_pdgm.png)
