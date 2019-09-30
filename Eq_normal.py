# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 11:49:01 2019

@author: cami_
"""

import numpy as np

def eq_normal(A,b):
    X  = np.dot(A.T, A)
    X2 = np.dot(A.T,b)
    return X,X2
    
