import numpy as np
from scipy.optimize import lsq_linear as ls

A = np.array([[-1,1,1],[-1,1,-1],[1,1,1],[-1,-1,1]]).astype(float)
b = np.array([1,-5,1,1]).astype(float)
print(ls(A,b))