import math
import numpy as np
import numpy

energies = [[24,      22,      30,      15,      18,      19],
            [12,      23,      15,      23,      10,     15],
            [11,      13,       22,      13,      21,      14],
            [13,      15,      17,      28,      19,      21],
            [17,      17,      7,       27,      20,      19]]

def minenergy(arr,row,prevcol,n,k):
    if row == len(arr):
        return 0
    C = len(arr[row])
    dp = [[math.inf for x in range(k+1)] for y in range(n)]
    if dp[row][prevcol+1] != math.inf:
        return dp[row][prevcol+1]
    res = math.inf
    for j in range(C):
        if j != prevcol:
            val = arr[row][j] + minenergy(arr,row+1,j,n,k)
            res = min(res,val)
    dp[row][prevcol+1] = res
    return res

def convertlistnumpy(list):
    return numpy.array([numpy.array(i) for i in list])

def minimum_seam(list):
    list = convertlistnumpy(list)
    r, c = len(list),len(list[0])

    M = list.copy()
    backtrack = np.zeros_like(M, dtype=np.int)

    for i in range(1, r):
        for j in range(0, c):
            if j == 0:
                index = np.argmin(M[i - 1, j:j + 2])
                backtrack[i, j] = index + j
                min_energy = M[i - 1, index + j]
            else:
                index = np.argmin(M[i - 1, j - 1:j + 2])
                backtrack[i, j] = index + j - 1
                min_energy = M[i - 1, index + j - 1]

            M[i, j] += min_energy
    a = np.argmin(M[len(M)-1])
    num = []
    for i in range(len(M[0]),0):
        num.append(M[i][a])
        a = backtrack[i][a]

    return M, backtrack