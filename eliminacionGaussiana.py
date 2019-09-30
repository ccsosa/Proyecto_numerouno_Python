def eliminacionGaussiana(a, etapas = False, mults = False):
    tol = 10**-14
    n = len (a) 
    variables = range(0,n)
    for k  in range (0,n-1):
        for i in range (k+1,n):
            multiplicador = a[i][k]/a[k][k]
            if mults:
                print "Mulitplicador de la fila ", i, ": ", multiplicador
            for j in range(k,n+1):
                valor = a[i][j] - (multiplicador * a[k][j])
                if abs(valor) < tol: 
                    valor = 0
                a[i][j] = valor
        if etapas:
            print "\nEtapa " + str(k+1) 
            imprimirMatriz(a, n)
    return a


def crearListaUnos(n):
    l = []
    for i in range(n):
        l.append(1)
    return l


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
                print a[i][j], " " * (lT[j] - len(str(a[i][j])) ), " ",
            else:
                print a[i][j], " ",
        print ""

def sustitucionRegresiva(a):
    n = len(a)
    variables = crearListaUnos(n)
    for i in range(n-1,-1,-1): #Crea una lista desde n hasta 0
        suma = 0
        for j in range(i+1,n):
            suma = suma + a[i][j]*variables[j]
        variables[i] = (a[i][n]-suma) / a[i][i]
    for x in range(n,0,-1):
        print "x" + str(x) + "= " + str(variables[x-1])



def main(a, etapas = False, mults = False):
    n = len(a)
    a = eliminacionGaussiana(a, etapas, mults)
    if not etapas:
        imprimirMatriz(a, n)
    sustitucionRegresiva(a)

