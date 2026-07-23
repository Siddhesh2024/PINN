from scipy.stats import qmc
import numpy as np

N_f = 10000  

# Latin Hypercube Sampling
sampler = qmc.LatinHypercube(d=2)
X_f = sampler.random(N_f)

x_f_train = -1 + 2*X_f[:,0:1]     
t_f_train = 0 + 1*X_f[:,1:2]
