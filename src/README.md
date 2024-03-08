# MSSD Bifiltration

Consider a list of $m$ trajectories of length $n$, $\{(x_i[1], x_i[2], ..., x_i[n])\}, i = 1,...,m$, such that each trajectory is a regular sampling of 
a set of flow lines on a topological space $X$. The objective of this project is to recover the topology of $X$ from this set of trajectories. The main 
tool we use here is the following bifiltration on the parameter set $\mathbb{R_+} \times [n]$, which we refer to as the *MSSD (Matching SubString Distance)
bifiltration*.

For each $(\epsilon, t) \in \mathbb{R_+}\times[n]$, define $\mathcal{G}(\epsilon, t)$ as the graph with vertices $[m]$, and an edge $(i, j)$ if the trajectories
$x_i[\cdot]$ and $x_j[\cdot]$ have subtrajectories of length greater than $t$ such that they stay within distance $\epsilon$, i.e
<p align="center">$[m] = V\left(\mathcal{G}(\epsilon, t) \right), \quad (i, j) \in E\left(\mathcal{G}(\epsilon, t) \right) \text{ if }\exists s_1, s_2$ such that $d(x_i[s_1 + s], x_j[s_2+s]) < \epsilon$ for $s=0,..,t-1$.</p>
 The MSSD bifiltration $X(\epsilon, t)$ is then defined as the Vietoris-Rips complex of $\mathcal{G}(\epsilon, t)$: $\quad X(\epsilon, t) = \mathcal{VR}(\mathcal{G}(\epsilon, t))$.

 Our proposition is that the Persistent homology of the MSSD bifiltraion contains significant information about the topology of $X$, and this is confirmed
 in the low dimensional example notebooks in [examples/](examples/).

## MSSD.py

The MSSD class in [MSSD.py](src/MSSD.py) contains the main algorithm that computes the necessary metric information for determining the MSSD bifiltration 
and its persistent homology. It takes as input two $n$ length trajectories: $(x_i[1],..,x_i[n])$ and $(x_j[1],...,x_j[n])$ and outputs an $n$ length vector
$(\epsilon[1], ..., \epsilon[n])$ such that for each $\epsilon > \epsilon[t], (i,j) \in E(\mathcal{G}(\epsilon, t))$. 