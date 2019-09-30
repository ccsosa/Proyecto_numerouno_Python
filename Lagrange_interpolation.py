# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 17:18:05 2019

@author: cami_
"""
import time
import numpy as np
import pandas as pd
import sympy as sym
import matplotlib.pyplot as plt
import functools
import operator
import math

training = pd.read_csv("E:/JAVERIANA/COMPUTACION/PROYECTO_UNO/data_to_int_training.csv")
test = pd.read_csv("E:/JAVERIANA/COMPUTACION/PROYECTO_UNO/data_to_int_testing.csv")

xi1 = training.iloc[:,0]
 #yi1 = np.log10(training.iloc[:,1])
yi1 = training.iloc[:,1]/1000

xi2 = test.iloc[:,0]
#yi2 = np.log10(test.iloc[:,1])
yi2 = test.iloc[:,1]/1000


#A = np.array([[-2,0,1], [-27,-1,0]])


#xi = np.array([0, 0.2, 0.3, 0.4])
#yi = np.array([1, 1.6, 1.7, 2.0])
#xi = np.array([-2,0,1])
#fi = np.array([-27,-1,0])

def int_lagrange(xi,fi,muestras):
# Interpolacion de Lagrange
# Polinomio en forma simbólica

# PROCEDIMIENTO
    n = len(xi)
    x = sym.Symbol('x')
    # Polinomio
    polinomio = 0
    for i in range(0,n,1):
        # Termino de Lagrange
        termino = 1
        for j  in range(0,n,1):
            if (j!=i):
                termino = termino*(x-xi[j])/(xi[i]-xi[j])
        polinomio = polinomio + termino*fi[i]
    # Expande el polinomio
    #px = polinomio.expand()
    # para evaluacion numérica
    pxn = sym.lambdify(x,polinomio)
    
    # Puntos para la gráfica
    a = np.min(xi)
    b = np.max(xi)
    
    xi_p = np.linspace(a,b,muestras)
    fi_p = pxn(xi_p)
    return(xi_p,fi_p,pxn)

"""
def interpolate(x, x_values, y_values):
    def _basis(j):
        p = [(x - x_values[m])/(x_values[j] - x_values[m]) for m in range(k) if m != j]
        return functools.reduce(operator.mul, p)
    assert len(x_values) != 0 and (len(x_values) == len(y_values)), 'x and y cannot be empty and must have the same length'
    k = len(x_values)
    return sum(_basis(j)*y_values[j] for j in range(k))


def int_lagrange(xi,yi,muestras):
    
    a = np.min(xi)
    b = np.max(xi)
    
    xi_p = np.linspace(a,b,muestras)
    
    XY = np.zeros([len(xi_p),2])
    
    for i in range(0,len(xi_p),1):
        XY[i,0] = xi_p[i]    
        XY[i,1] = interpolate(x=xi_p[i], x_values=xi, y_values=yi)
    X= np.array(XY[:,0])
    Y= np.array(XY[:,1])
    return(X,Y)
    
"""



    

#X = int_lagrange(xi,yi,101)
#X2 = int_lagrange(xi2,yi2,101)

time_start = time.perf_counter()
X1 = int_lagrange(xi=xi1,fi=yi1,muestras=101)
time_elapsed = (time.perf_counter() - time_start)


time_start = time.perf_counter()
X2 = int_lagrange(xi=xi2,fi=yi2,muestras=101)
time_elapsed2 = (time.perf_counter() - time_start)


plt.title('Interpolación de Lagrange (datos de entrenamiento)')
plt.plot(xi1,yi1,'o', label = 'Puntos')
plt.plot(X1[0].T,X1[1].T, label = 'Polinomio')
plt.ylim((-5, 2000)) 
plt.xlabel("Temperatura media anual (℃)")
plt.ylabel("Precipitación media anual (mm)")
plt.legend()
plt.show()

#######################


plt.title('Interpolación de Lagrange (datos de evaluación)')
plt.plot(xi2,yi2,'o', label = 'Puntos')
plt.plot(X2[0].T,X2[1].T, label = 'Polinomio')
plt.ylim((-5, 2000)) 
plt.xlabel("Temperatura media anual (℃)")
plt.ylabel("Precipitación media anual (mm)")
plt.legend()
plt.show()

