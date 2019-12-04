import numpy as np 
import math 
from utils import DLU,plot_x
import time
import matplotlib.pyplot as plt
import scipy.sparse.linalg

def optimal_w(A) :
    
    D,L,U = DLU(A)
    R = L+U
    T = np.matmul(np.linalg.inv(D), (R))
    eig,_ = np.linalg.eig(T)
    print(f"eigen values {eig}")
    w = 2 / (1 + np.sqrt(1 - np.abs(eig).max()**2   ) )
    return w

def SOR(A,b,x_prev,tol,maxiter,w) :
    
    iteration = 0
    rel_diff = tol * 2

    if x_prev is None:
        x_prev = np.zeros(len(A[0]))
    all_x = [x_prev]


    D,L,U = DLU(A)

    T = np.matmul(np.linalg.inv(D - w*L), (1-w) * D + w * U)
    C = np.matmul(np.linalg.inv(D - w*L), w*b)

    eig,_ = np.linalg.eig(T)
    spectral_radius = np.abs(eig).max()


    start_time = time.time()
    while (rel_diff > tol) and (iteration < maxiter):
        
        iteration+=1
        
        x = np.matmul(T,x_prev) + C
        
        
        rel_diff = np.linalg.norm(x - x_prev) / np.linalg.norm(x)
        all_x.append(x)
        x_prev = x


        # if(np.linalg.norm(np.matmul(A,x) - b) > error):
        #     # if(iteration % 1000 == 0):
        #     print('Iteration=',iteration, 'error=', np.linalg.norm(np.matmul(A,x) - b))     
        #     continue
        # else :
        #     break

    return x,iteration,time.time() - start_time, all_x,spectral_radius


# x = SOR(A, b, x0, tol, max_iter, w)
# x,iteration,total_time, all_x = SOR2(A, b, np.array([0,0]), tol, max_iter, w)
# plot_x(all_x)

# print( x, iteration, total_time )


if __name__ == "__main__":
    X = [-1,0,0,0]
    y = 0
    
    ''' Find valid A and b '''
    while(not(all(x >= 0 for x in X))):
        y = y+1
        #print(y)
        dim = 10
        a = np.random.randint(-10,10, size=(dim,dim))
        A = (a + a.T)/2
        for i in range(dim):
            if((y+i)%3 == 0):
                A[i][i] = np.random.randint(100,200)
            else :
                A[i][i] = -(np.random.randint(100,200))
        I = np.linalg.inv(A)
        b = np.random.randint(-1000,1000, size=dim)
        X = np.matmul(I,b)
    print(A)
    print(b)
    # SOR(A,b,X,1E-15,200,1.5)
    x,iteration,total_time, all_x, spectral_radius = SOR(A,b,np.zeros(b.shape),1E-15,200,optimal_w(A))
    print(x)
    print(iteration)
    print(total_time)
    # print(all_x)
    # plot_x(all_x)

    