
import sys
import os
import numpy as np
import sympy as sym
import matplotlib.pyplot as plt
import pandas as pd
import time
import random 
import statistics

sys.path.append(os.path.abspath("E:/JAVERIANA/COMPUTACION/Gauss.py"))
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

xi = training.iloc[:,0]
yi = training.iloc[:,1]

xi2 = test.iloc[:,0]
yi2 = test.iloc[:,1]

#INTERPOLACION DE NEWTON USANDO DIFERENCIAS DIVIDIDAS

def diferencias_divididas(x, lx, ly):
    """Metodo numerico de dfierencias dividadas 
    
    Arguments:
    - `x`: Valor a interpolar
    - `lx`: Lista con los valores de x
    - `ly`: Lista con los valores de y (f(x))
    """
    y = 0
    for i in range(len(lx)-1):
        if x >= lx[i] and x <= lx[i+1]:
            y = (ly[i+1] - ly[i]) / (lx[i+1]-lx[i]) * (x - lx[i]) + ly[i]
    return y  

      
##################################################      
#EVALUANDO EL METODO POR INTERVALOS

def int_newton(xi,yi,muestras):
    
    a = np.min(xi)
    b = np.max(xi)
    
    xi_p = np.linspace(a,b,muestras)
    
    XY = np.zeros([len(xi_p),2])
    
    for i in range(0,len(xi_p),1):
        XY[i,0] = xi_p[i]    
        XY[i,1] = diferencias_divididas(x=xi_p[i], lx=xi, ly=yi)
    X= np.array(XY[:,0])
    Y= np.array(XY[:,1])
    return(X,Y)
    
    
##################################################
 #Probando la interpolacion
X = int_newton(xi,yi,101)
X2 = int_newton(xi2,yi2,101)

##################################################
 #Tiempo de computo
time_start = time.perf_counter()
X1 = int_newton(xi,yi,100)
time_elapsed = (time.perf_counter() - time_start)

time_start = time.perf_counter()
X2 = int_newton(xi2,yi2,100)
time_elapsed2 = (time.perf_counter() - time_start)


#######################
#plot1
plt.title('Interpolación de Newton (datos de entrenamiento)')
plt.plot(xi,yi,'o', label = 'Puntos')
plt.plot(X[0].T,X[1].T, label = 'Polinomio')
plt.ylim((200, 1400)) 
plt.xlabel("Temperatura media anual (℃)")
plt.ylabel("Precipitación media anual (mm)")
plt.legend()
plt.show()

#######################
#plot 2

plt.title('Interpolación de Newton (datos de evaluación)')
plt.plot(xi2,yi2,'o', label = 'Puntos')
plt.plot(X2[0].T,X2[1].T, label = 'Polinomio')
plt.xlabel("Temperatura media anual (℃)")
plt.ylabel("Precipitación media anual (mm)")
plt.legend()
plt.show()


#######################
#RMSE
rms_data(X1[1].T,X2[1].T)

RM = []
for i in range(0,99,1):
    x_rms = rms_data(yi,random.sample(list(X1[1]), k=50))
    RM.append(x_rms)
    
x_RM = statistics.mean(RM)    
x_RM    

RM2 = []
for i in range(0,99,1):
    x_rms = rms_data(yi2,random.sample(list(X2[1]), k=50))
    RM2.append(x_rms)
    
x_RM2 = statistics.mean(RM2)    
x_RM2    