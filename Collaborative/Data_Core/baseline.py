# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 16:37:58 2021

@author: nunoa
"""

import numpy as np
from scipy import *
from scipy import sparse
from scipy.sparse.linalg import spsolve

#assymetric least squares baseline removal method
def baseline_als(y, lam, p, niter=10):
  L = len(y)
  D = sparse.diags([1,-2,1],[0,-1,-2], shape=(L,L-2))
  w = np.ones(L)
  for i in range(niter):
    W = sparse.spdiags(w, 0, L, L)
    Z = W + lam * D.dot(D.transpose())
    z = spsolve(Z, w*y)
    w = p * (y > z) + (1-p) * (y < z)
  return z
