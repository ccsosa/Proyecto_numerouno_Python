# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 22:37:35 2019

@author: cami_
"""
import sys
import os
import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath("E:/JAVERIANA/COMPUTACION/LUGauss"))
sys.path.append(os.path.abspath("E:/JAVERIANA/COMPUTACION/Householder"))
sys.path.append(os.path.abspath("E:/JAVERIANA/COMPUTACION/Eq_normal"))
from LUGauss import*
from Householder import*
from Eq_normal import*

training = pd.read_csv("E:/JAVERIANA/COMPUTACION/PROYECTO_UNO/data_to_int_training.csv")
test = pd.read_csv("E:/JAVERIANA/COMPUTACION/PROYECTO_UNO/data_to_int_testing.csv")

Xi = np.array([-1,-0.5,0,0.5,1],dtype=float)
Yi = np.array([1,0.5,0,0.5,2],dtype=float)



def lineal_least_squares(Xi,Yi):
    mat = np.zeros((len(Xi),3),dtype=float)
    mat[:,0] = 1
    mat[:,1] = Xi
    mat[:,2] = np.multiply(Xi,Xi)

    AB = copy.deepcopy(mat);AB_o =  AB

    x = eq_normal(AB_o,Yi)
    x_chol = np.linalg.cholesky(x[0])
    ly_to_solve = copy.deepcopy(x_chol)  

    ly_to_solve = np.column_stack([ly_to_solve, x[1]])
    y=LUGauss(ly_to_solve)

    LTx_to_solve = copy.deepcopy(x_chol.T)
    LTx_to_solve = np.column_stack([LTx_to_solve, y[3]])
  
    x=LUGauss(LTx_to_solve)

    x_par = x[3]

    return(x_par)

 
 
def lls_plot(Xi,muestras,fun_lls):
    
    a = np.min(Xi)
    b = np.max(Xi)
    xi_p = np.linspace(a,b,muestras)
     
    yi_p = fun_lls[0]+(fun_lls[1]*xi_p)+(fun_lls[2]*(xi_p*xi_p))
   
    XY = np.zeros((len(xi_p),2))
    XY[:,0] = xi_p
    XY[:,1] = yi_p
    return(XY)

fun_lls =  lineal_least_squares(Xi,Yi) 
XY = lls_plot(Xi,101,fun_lls)
XY = np.matrix(XY)

#####

xi1 = training.iloc[:,0]
yi1 = training.iloc[:,1]

fun_lls =  lineal_least_squares(xi1,yi1)
XY = lls_plot(xi1,101,fun_lls)
XY = np.matrix(XY)

#####
plt.title('Mínimos cuadrados (datos de entrenamiento)')
plt.plot(xi1,yi1,'o', label = 'Puntos')
plt.plot(XY[:,0],XY[:,1], label = 'Polinomio')
plt.xlabel("Temperatura media anual (℃)")
plt.ylabel("Precipitación media anual (mm)")
plt.legend()
plt.show()

#####

xi2 = test.iloc[:,0]
yi2 = test.iloc[:,1]

fun_lls =  lineal_least_squares(xi2,yi2)
XY = lls_plot(xi1,101,fun_lls)
XY = np.matrix(XY)

#####
plt.title('Mínimos cuadrado (datos de evaluación)')
plt.plot(xi2,yi2,'o', label = 'Puntos')
plt.plot(XY[:,0],XY[:,1], label = 'Polinomio')
plt.xlabel("Temperatura media anual (℃)")
plt.ylabel("Precipitación media anual (mm)")
plt.legend()
plt.show()


