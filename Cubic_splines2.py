
import numpy as np
import sympy as sym
import sys
import os
from math import sqrt
from pprint import pprint
import numpy as np
import copy
import pandas as pd
import sympy as sym
import matplotlib.pyplot as plt
import random
import statistics
import time

sys.path.append(os.path.abspath("E:/JAVERIANA/COMPUTACION/RMS"))

from RMS import*

def spline_cubic_nat(xi,yi):
    n = len(xi)
    
    #h
    h = np.zeros(n-1, dtype = float)
    for j in range(0,n-1,1):
        h[j] = xi[j+1] - xi[j]
    
    # Ecuaciones
    A = np.zeros(shape=(n-2,n-2), dtype = float)
    B = np.zeros(n-2, dtype = float)
    S = np.zeros(n, dtype = float)
    A[0,0] = 2*(h[0]+h[1])
    A[0,1] = h[1]
    B[0] = 6*((yi[2]-yi[1])/h[1] - (yi[1]-yi[0])/h[0])
    for i in range(1,n-3,1):
        A[i,i-1] = h[i]
        A[i,i] = 2*(h[i]+h[i+1])
        A[i,i+1] = h[i+1]
        B[i] = 6*((yi[i+2]-yi[i+1])/h[i+1] - (yi[i+1]-yi[i])/h[i])
    A[n-3,n-4] = h[n-3]
    A[n-3,n-3] = 2*(h[n-3]+h[n-2])
    B[n-3] = 6*((yi[n-1]-yi[n-2])/h[n-2] - (yi[n-2]-yi[n-3])/h[n-3])
    
    # Resolver sistema de ecuaciones


    r = np.linalg.solve(A,B)
    # S
    for j in range(1,n-1,1):
        S[j] = r[j-1]
    S[0] = 0
    S[n-1] = 0
    
    # Coeficientes
    a = np.zeros(n-1, dtype = float)  #a1
    b = np.zeros(n-1, dtype = float) #a2
    c = np.zeros(n-1, dtype = float) #a3
    d = np.zeros(n-1, dtype = float) #a4
    for j in range(0,n-1,1):
        a[j] = (S[j+1]-S[j])/(6*h[j])
        b[j] = S[j]/2
        c[j] = (yi[j+1]-yi[j])/h[j] - (2*h[j]*S[j]+h[j]*S[j+1])/6
        d[j] = yi[j]
    
    # Polinomio cuadratico
    x = sym.Symbol('x')
    polinomio = []
    for j in range(0,n-1,1):
        ptramo = a[j]*(x-xi[j])**3 + b[j]*(x-xi[j])**2 + c[j]*(x-xi[j])+ d[j]
        ptramo = ptramo.expand()
        polinomio.append(ptramo)
    
    return(polinomio)


training = pd.read_csv("E:/JAVERIANA/COMPUTACION/PROYECTO_UNO/data_to_int_training.csv")
test = pd.read_csv("E:/JAVERIANA/COMPUTACION/PROYECTO_UNO/data_to_int_testing.csv")
    #A = np.array([[-2,0,1], [-27,-1,0]])
training = training.sort_values("bio_1")
test = test.sort_values("bio_1")

xi = np.array(training.iloc[:,0])
yi = np.array(training.iloc[:,1])

xi2 = np.array(test.iloc[:,0])
yi2 = np.array(test.iloc[:,1])

# Obtiene los polinomios por tramos
polinomio1 = spline_cubic_nat(xi,yi)
polinomio2 = spline_cubic_nat(xi2,yi2)


# GRAFICA
# Puntos para grafica en cada tramo
n = len(xi)
tramo = 1
resolucion=100

def cubic_eval(polinomio,tramo,resolucion):   
    xtrazado = np.array([])
    ytrazado = np.array([])

    while not(tramo>=n):
        a = xi[tramo-1]
        b = xi[tramo]
        xtramo = np.linspace(a,b,resolucion)
    
        ptramo = polinomio[tramo-1]
        pxtramo = sym.lambdify('x',ptramo)
        ytramo = pxtramo(xtramo)
    
        xtrazado = np.concatenate((xtrazado,xtramo))
        ytrazado = np.concatenate((ytrazado,ytramo))
        tramo = tramo + 1

    return(xtrazado,ytrazado)

cubic_eval_df1 = cubic_eval(polinomio1,tramo,resolucion)
cubic_eval_df2 = cubic_eval(polinomio2,tramo,resolucion)

#######################
#PLOT

plt.title('Spline cúbico natural (datos de entrenamiento)')
plt.plot(xi,yi,'o', label = 'Puntos')
plt.plot(cubic_eval_df1[0],cubic_eval_df1[1], label = 'Polinomio')
plt.xlabel("Temperatura media anual (℃)")
plt.ylabel("Precipitación media anual (mm)")
plt.ylim(min(yi)-300,max(yi)+300)
plt.legend()
plt.show()


plt.title('Spline cúbico natural (datos de evaluación)')
plt.plot(xi2,yi2,'o', label = 'Puntos')
plt.plot(cubic_eval_df2[0],cubic_eval_df2[1], label = 'Polinomio')
plt.xlabel("Temperatura media anual (℃)")
plt.ylabel("Precipitación media anual (mm)")
plt.ylim(min(yi)-300,max(yi)+300)
plt.legend()
plt.show()



####################### 
 #Tiempo de computo
time_start = time.perf_counter()
cubic_eval_df1 = cubic_eval(polinomio1,tramo,resolucion)
time_elapsed = (time.perf_counter() - time_start)

time_start = time.perf_counter()
cubic_eval_df2 = cubic_eval(polinomio2,tramo,resolucion)
time_elapsed2 = (time.perf_counter() - time_start)



#######################
#RMSE
rms_data(cubic_eval_df1[1],cubic_eval_df2[1])

RM = []
for i in range(0,99,1):
    x_rms = rms_data(yi,random.sample(list(cubic_eval_df1[1]), k=50))
    RM.append(x_rms)
    
x_RM = statistics.mean(RM)    
x_RM    

RM2 = []
for i in range(0,99,1):
    x_rms = rms_data(yi2,random.sample(list(cubic_eval_df2[1]), k=50))
    RM2.append(x_rms)
    
x_RM2 = statistics.mean(RM2)    
x_RM2    