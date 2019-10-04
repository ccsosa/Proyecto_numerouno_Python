
import time
import sys
import os
import numpy as np
import pandas as pd
import sympy as sym
import matplotlib.pyplot as plt
import functools
import operator
import math
from random import sample 
import random 
import statistics

sys.path.append(os.path.abspath("E:/JAVERIANA/COMPUTACION/RMS"))

from RMS import*

training = pd.read_csv("E:/JAVERIANA/COMPUTACION/PROYECTO_UNO/data_to_int_training.csv")
test = pd.read_csv("E:/JAVERIANA/COMPUTACION/PROYECTO_UNO/data_to_int_testing.csv")

training = training.sort_values("bio_1")
test = test.sort_values("bio_1")

xi1 = training.iloc[:,0]
yi1 = training.iloc[:,1]

xi2 = test.iloc[:,0]
yi2 = test.iloc[:,1]

def int_lagrange(xi,yi,muestras):
    
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
        polinomio = polinomio + termino*yi[i]
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


##################################################      
#EVALUANDO EL METODO POR INTERVALOS
 #Tiempo de computo
time_start = time.perf_counter()
X1 = int_lagrange(xi=xi1,yi=yi1,muestras=100)
time_elapsed = (time.perf_counter() - time_start)


time_start = time.perf_counter()
X2 = int_lagrange(xi=xi2,yi=yi2,muestras=100)
time_elapsed2 = (time.perf_counter() - time_start)

##################################################      
#PLOT

plt.title('Interpolación de Lagrange (datos de entrenamiento)')
plt.plot(xi1,yi1,'o', label = 'Puntos')
plt.plot(X1[0].T,X1[1].T, label = 'Polinomio')
#plt.ylim((-5, 2000)) 
plt.xlabel("Temperatura media anual (℃)")
plt.ylabel("Precipitación media anual (mm)")
plt.legend()
plt.show()

#######################

plt.title('Interpolación de Lagrange (datos de evaluación)')
plt.plot(xi2,yi2,'o', label = 'Puntos')
plt.plot(X2[0].T,X2[1].T, label = 'Polinomio')
#plt.ylim((-5, 2000)) 
plt.xlabel("Temperatura media anual (℃)")
plt.ylabel("Precipitación media anual (mm)")
plt.legend()
plt.show()

#######################
#RMSE


rms_data(X1[1],X2[1])

RM = []
for i in range(0,99,1):
    x_rms = rms_data(yi1,random.sample(list(X1[1]), k=50))
    RM.append(x_rms)
    
x_RM = statistics.mean(RM)    
x_RM    

RM2 = []
for i in range(0,99,1):
    x_rms = rms_data(yi2,random.sample(list(X2[1]), k=50))
    RM2.append(x_rms)
    
x_RM2 = statistics.mean(RM2)    
x_RM2    