
import sys
import os
import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt
import time

sys.path.append(os.path.abspath("E:/JAVERIANA/COMPUTACION/LUGauss"))
sys.path.append(os.path.abspath("E:/JAVERIANA/COMPUTACION/Householder"))
sys.path.append(os.path.abspath("E:/JAVERIANA/COMPUTACION/Eq_normal"))
sys.path.append(os.path.abspath("E:/JAVERIANA/COMPUTACION/RMS"))
sys.path.append(os.path.abspath("E:/JAVERIANA/COMPUTACION/Gram_schmidt"))
from LUGauss import*
from Householder import*
from Eq_normal import*
from RMS import*
from Gram_schmidt import*

os.chdir("E:/JAVERIANA/COMPUTACION")
training = pd.read_csv("E:/JAVERIANA/COMPUTACION/PROYECTO_UNO/data_to_int_training.csv")
test = pd.read_csv("E:/JAVERIANA/COMPUTACION/PROYECTO_UNO/data_to_int_testing.csv")

training = training.sort_values("bio_1")
test = test.sort_values("bio_1")

"""
Xi = np.array([-1,-0.5,0,0.5,1],dtype=float)
Yi = np.array([1,0.5,0,0.5,2],dtype=float)
"""

## LEAST SQUARES ECUACIONES NORMALES
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

## LEAST SQUARES HOUSEHOLDER
def lineal_least_squares_HOUSE(Xi,Yi):
    mat = np.zeros((len(Xi),3),dtype=float)
    mat[:,0] = 1
    mat[:,1] = Xi
    mat[:,2] = np.multiply(Xi,Xi)

    Ab = copy.deepcopy(mat)
    Ab = np.column_stack([Ab, Yi])
    X= qr_householder(Ab)
    

    R_to_solve = copy.deepcopy(X[1])
    
    n = R_to_solve.shape[1]
    a = R_to_solve[:,range(0,(n-1),1)]
    b = R_to_solve[:,n-1]
   
    x_par = sustitucionProgresiva(a,b,n-1)
 
    return(x_par)

## LEAST SQUARES GRAM-SCHMIDT
def lineal_least_squares_GRAM(Xi,Yi):
    mat = np.zeros((len(Xi),3),dtype=float)
    mat[:,0] = 1
    mat[:,1] = Xi
    mat[:,2] = np.multiply(Xi,Xi)

    Ab = copy.deepcopy(mat)
    Ab = np.column_stack([Ab, Yi])
    X= qr_householder(Ab)
    

    R_to_solve = copy.deepcopy(X[1])
    
    n = R_to_solve.shape[1]
    a = R_to_solve[:,range(0,(n-1),1)]
    b = R_to_solve[:,n-1]
   
    x_par = sustitucionProgresiva(a,b,n-1)
 
    return(x_par)
    
 ## EVALUANDO FUNCION
def lls_plot(Xi,muestras,fun_lls):
    
    a = np.min(Xi)
    b = np.max(Xi)
    xi_p = np.linspace(a,b,muestras)
    yi_p = fun_lls[0]+(fun_lls[1]*xi_p)+(fun_lls[2]*(xi_p*xi_p))
   
    XY = np.zeros((len(xi_p),2))
    XY[:,0] = xi_p
    XY[:,1] = yi_p
    return(XY)


#####TRAINING
xi1 = training.iloc[:,0]
yi1 = training.iloc[:,1]

#####LS TRAINING
fun_lls =  lineal_least_squares(xi1,yi1) 
XY = lls_plot(xi1,100,fun_lls)
XY = np.matrix(XY)

#####PLOT TRAINING
plt.title('Mínimos cuadrados (datos de entrenamiento)')
plt.plot(xi1,yi1,'o', label = 'Puntos')
plt.plot(XY[:,0],XY[:,1], label = 'Polinomio')
plt.xlabel("Temperatura media anual (℃)")
plt.ylabel("Precipitación media anual (mm)")
plt.legend()
plt.show()

#####TEST
xi2 = test.iloc[:,0]
yi2 = test.iloc[:,1]

#####LS TRAINING
fun_lls =  lineal_least_squares(xi2,yi2)
XY = lls_plot(xi1,100,fun_lls)
XY = np.matrix(XY)

#####PLOT TESTING
plt.title('Mínimos cuadrado (datos de evaluación)')
plt.plot(xi2,yi2,'o', label = 'Puntos')
plt.plot(XY[:,0],XY[:,1], label = 'Polinomio')
plt.xlabel("Temperatura media anual (℃)")
plt.ylabel("Precipitación media anual (mm)")
plt.legend()
plt.show()

##TIME
time_start = time.perf_counter()
fun_lls =  lineal_least_squares(xi1,yi1) 
XY1a = lls_plot(xi1,100,fun_lls)
time_elapsed1A = (time.perf_counter() - time_start)

time_start = time.perf_counter()
fun_lls =  lineal_least_squares_HOUSE(xi1,yi1)
XY1b = lls_plot(xi1,100,fun_lls)
time_elapsed2A = (time.perf_counter() - time_start)

time_start = time.perf_counter()
fun_lls =  lineal_least_squares_GRAM(xi1,yi1)
XY1c = lls_plot(xi1,100,fun_lls)
time_elapsed3A = (time.perf_counter() - time_start)


##################################################

#####Tiempo de computo
time_start = time.perf_counter()
fun_lls =  lineal_least_squares(xi2,yi2) 
XY2a = lls_plot(xi2,100,fun_lls)
time_elapsed1B = (time.perf_counter() - time_start)

time_start = time.perf_counter()
fun_lls =  lineal_least_squares_HOUSE(xi2,yi2) 
XY2b = lls_plot(xi2,100,fun_lls)
time_elapsed2B = (time.perf_counter() - time_start)


time_start = time.perf_counter()
fun_lls =  lineal_least_squares_GRAM(xi2,yi2)
XY2c = lls_plot(xi2,100,fun_lls)
time_elapsed3B = (time.perf_counter() - time_start)

#######################
#RMSE
rms_data(XY1a[:,0],XY1a[:,0])
rms_data(XY1a[:,0],XY1c[:,0])


rms_data(XY2a[:,0],XY2b[:,0])
rms_data(XY2a[:,0],XY2c[:,0])


rms_data(XY1a[:,0],XY2a[:,0])
rms_data(XY1b[:,0],XY2b[:,0])
rms_data(XY1c[:,0],XY2c[:,0])