# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 17:06:54 2019

@author: cami_
"""
import sys
import os
from math import sqrt
from pprint import pprint
import numpy as np
import copy
sys.path.append(os.path.abspath("E:/JAVERIANA/COMPUTACION/LUGauss"))
sys.path.append(os.path.abspath("E:/JAVERIANA/COMPUTACION/Householder"))
sys.path.append(os.path.abspath("E:/JAVERIANA/COMPUTACION/Eq_normal"))
from LUGauss import*
from Eq_normal import*

def qr_householder(A):
    m, n = A.shape
    Q = np.eye(m) # Orthogonal transform so far
    R = A.copy() # Transformed matrix so far

    for j in range(n):
        # Find H = I - beta*u*u' to put zeros below R[j,j]
        x = R[j:, j]
        normx = np.linalg.norm(x)
        rho = -np.sign(x[0])
        u1 = x[0] - rho * normx
        u = x / u1
        u[0] = 1
        beta = -rho * u1 / normx

        R[j:, :] = R[j:, :] - beta * np.outer(u, u).dot(R[j:, :])
        Q[:, j:] = Q[:, j:] - beta * Q[:, j:].dot(np.outer(u, u))
        
    return Q, R
"""
A = np.array([[1,-1,1],
              [1,-0.5,0.25],
              [1,0,0],
              [1,0.5,0.25],
              [1,1,1]])

b = np.array([1,0.5,0,0.5,2])

Ab = copy.deepcopy(A)
Ab = np.column_stack([Ab, b])
X= qr_householder(Ab)

QR = np.dot(X[0],X[1])
QR_to_solve = copy.deepcopy(QR)
QR_to_solve = np.column_stack([X[0], b])

QR = np.dot(X[0],X[1])
x=LUGauss(QR_to_solve)
"""