
import numpy as np

def crearL(n):
    L = []
    for i in range(n):
        fila = []
        for j in range(n):
            if i == j:
                fila.append(1.0)
            else:
                fila.append(0)
        L.append(fila)
    return L

def sacarB(a, n):
    b = []
    for i in range(n):
        b.append(a[i][n])
    return b

def eliminacionGaussiana(a, L, etapas = False, mults = False):
    tol = 10**-14
    n = len (a) 
    variables = range(0,n)
    for k  in range (0,n-1):
        for i in range (k+1,n):
            L[i][k] = a[i][k]/a[k][k]
            if mults:
                print("Mulitplicador de la fila ", i, ": ", L[i][k])
            for j in range(k,n+1):
                valor = a[i][j] - (L[i][k] * a[k][j])
                if abs(valor) < tol: 
                    valor = 0
                a[i][j] = valor
        if etapas:
            print("\nEtapa " + str(k+1))
            imprimirMatriz(a, n)
    return (a, L)


def crearListaUnos(n):
    l = []
    for i in range(n):
        l.append(1)
    return l

def sustitucionRegresiva(a, b, n):
    variables = crearListaUnos(n)
    for i in range(n-1,-1,-1): #Crea una lista desde n hasta 0
        suma = 0
        for j in range(i+1,n):
            suma = suma + a[i][j]*variables[j]
        variables[i] = (b[i]-suma) / a[i][i]
    return variables

def sustitucionProgresiva(a, b, n):
    variables = crearListaUnos(n)
    for i in range(n):
        suma = 0
        for j in range(i):
            suma = suma + a[i][j]*variables[j]
        variables[i] = (b[i]-suma) / a[i][i]
    return variables


def busqueda(a, j, n):
    mayor = 0
    for i in range (0, n):
        tam = len(str(a[i][j]))
        if mayor < tam:
            mayor = tam
    return mayor


def imprimirMatriz(a, n):
    lT = []
    for i in range (0, n):
        for j in range (0, n+1):
            if len(lT) <= j:
                lT.append( busqueda(a, j, n) )
            if lT[j] > len(str(a[i][j])):
                print(a[i][j], " " * (lT[j] - len(str(a[i][j])) ), " ")
            else:
                print(a[i][j], " ")
        print("")

def imprimirMatrizCuadrada(a, n):
    lT = []
    for i in range (0, n):
        for j in range (0, n):
            if len(lT) <= j:
                lT.append( busqueda(a, j, n) )
            if lT[j] > len(str(a[i][j])):
                print(a[i][j], " " * (lT[j] - len(str(a[i][j])) ), " ")
            else:
                print(a[i][j], " ")
        print("")


def LUGauss(a, etapas = False, mults=False):
    n = len(a)
    L = crearL(n)    
    b = sacarB(a, n)
    a, L = eliminacionGaussiana(a, L, etapas, mults)
    
    #print("Matriz L: ")
    imprimirMatrizCuadrada(L,n)
    #print("Matriz A: ")
    imprimirMatrizCuadrada(a,n)
    z = sustitucionProgresiva(L,b,n)
    x = sustitucionRegresiva(a,z,n)
#    for i in range(n):
 #       print("x" + str(i+1), ": ", x[i]
    return(a,L,z,x)
"""
A = np.array([[2,4,-2],
              [4,9,-3],
              [-2,-3,7]
              ])

B = np.array([[2],
              [8],
              [10]])
    
    
AB = np.concatenate((A,B), axis=1)



x = LUGauss(a=AB, etapas = False, mults=False)

"""