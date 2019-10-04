
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


training = pd.read_csv("E:/JAVERIANA/COMPUTACION/PROYECTO_UNO/data_to_int_training.csv")
test = pd.read_csv("E:/JAVERIANA/COMPUTACION/PROYECTO_UNO/data_to_int_testing.csv")
    #A = np.array([[-2,0,1], [-27,-1,0]])
training = training.sort_values("bio_1")
test = test.sort_values("bio_1")

#   xi = np.array([0, 0.2, 0.3, 0.4])
#yi = np.array([1, 1.6, 1.7, 2.0])

#xi = np.array([-2,0,1])
#yi = np.array([-27,-1,0]) 

xi = training.iloc[:,0];xi = np.array(xi)
yi = training.iloc[:,1];yi = np.array(yi)

xi2 = test.iloc[:,0];xi2 = np.array(xi2)
yi2 = test.iloc[:,1];yi2 = np.array(yi2)


def spline_lineal(lx, ly):
    """Metodo numerico de spline Lineal
    
    Arguments:
    - `lx`: Lista con los valores de x
    - `ly`: Lista con los valores de y
    """
    fx = []
    for i in range(len(lx)-1):
        fx.append( (str(ly[i]) + " + " + str((ly[i] - ly[i+1]) / (lx[i] - lx[i+1])) + "*(x - " + str( lx[i] ) + ")").replace("+ -", "-"))
    return fx

#######################
#probar función
muestras=10



def spline_test(ff_list,muestras):
    
    XY = []
    for i in range(0,len(xi)-1):
        print(i)
        
        x = sym.Symbol('x')
        polinomio = ff_list[i]
        pxn = sym.lambdify(x,polinomio)
   
        xi_to = [xi[i],xi[i+1]]
        a = np.min(xi_to)
        b = np.max(xi_to)
    
        xi_p = np.linspace(a,b,muestras)
        xy_to = np.zeros([(len(xi_p)),4])
        fi_p = pxn(xi_p)
        
        xy_to[:,0] = a
        xy_to[:,1] = b
        xy_to[:,2] = xi_p
        xy_to[:,3] = fi_p
        XY.append(xy_to)
    return(XY)

ff_list1 = spline_lineal(xi,yi)
ff_list2 = spline_lineal(xi2,yi2)

xx1 = spline_test(ff_list1,muestras)
xx1 = np.concatenate(xx1)
xx2 = spline_test(ff_list2,muestras)
xx2 = np.concatenate(xx2)
#######################
#PLOT

plt.title('Spline lineal natural (datos de entrenamiento)')
plt.plot(xi,yi,'o', label = 'Puntos')
plt.plot(xx1[:,2],xx1[:,3], label = 'Polinomio')
plt.xlabel("Temperatura media anual (℃)")
plt.ylabel("Precipitación media anual (mm)")
plt.ylim(min(yi2)-300,max(yi2)+300)
plt.legend()
plt.show()

plt.title('Spline lineal natural (datos de evaluación)')
plt.plot(xi2,yi2,'o', label = 'Puntos')
plt.plot(xx2[:,2],xx2[:,3], label = 'Polinomio')
plt.xlabel("Temperatura media anual (℃)")
plt.ylabel("Precipitación media anual (mm)")
plt.ylim(min(yi2)-300,max(yi2)+300)
plt.legend()
plt.show()

####################### 
 #Tiempo de computo
time_start = time.perf_counter()
ff_list1 = spline_lineal(xi,yi)
time_elapsed = (time.perf_counter() - time_start)

time_start = time.perf_counter()
ff_list2 = spline_lineal(xi2,yi2)
time_elapsed2 = (time.perf_counter() - time_start)



#######################
#RMSE
rms_data(xx1[:,3],xx2[:,3])

RM = []
for i in range(0,99,1):
    x_rms = rms_data(yi,random.sample(list(xx1[:,3]), k=50))
    RM.append(x_rms)
    
x_RM = statistics.mean(RM)    
x_RM    

RM2 = []
for i in range(0,99,1):
    x_rms = rms_data(yi2,random.sample(list(xx2[:,3]), k=50))
    RM2.append(x_rms)
    
x_RM2 = statistics.mean(RM2)    
x_RM2    
