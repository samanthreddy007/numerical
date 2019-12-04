import numpy as np
from utils import DLU,plot_x
import time

def jacobi2(A, b, x0, tol, maxiter=200):
    """
    Performs Jacobi iterations to solve the line system of
    equations, Ax=b, starting from an initial guess, ``x0``.

    Terminates when the change in x is less than ``tol``, or
    if ``maxiter`` [default=200] iterations have been exceeded.

    Returns 3 variables:
        1.  x, the estimated solution
        2.  rel_diff, the relative difference between last 2
            iterations for x
        3.  k, the number of iterations used.  If k=maxiter,
            then the required tolerance was not met.
    """
    n = A.shape[0]
    x = x0.copy()
    x_prev = x0.copy()
    k = 0
    rel_diff = tol * 2

    while (rel_diff > tol) and (k < maxiter):
        for i in range(0, n):
            subs = 0.0
            for j in range(0, n):
                if i != j:
                    subs += A[i,j] * x_prev[j]
        

            x[i] = (b[i] - subs ) / A[i,i]
        k += 1

        rel_diff = np.linalg.norm(x - x_prev) / np.linalg.norm(x)
        # print(x, rel_diff)
        x_prev = x.copy()

    return x, rel_diff, k


def jacobi(A, b, x_prev,tol, maxiter):

    iteration = 0
    rel_diff = tol * 2
    if x_prev is None:
        x_prev = np.zeros(len(A[0]))
    all_x = [x_prev]
                                                                                                                                                                
    D,L,U = DLU(A)
    R = L+U

    # print(f"D-1 {np.linalg.inv(D)}")
    # print(f"R {R}")
    T = np.matmul(np.linalg.inv(D), (R))
    C = np.matmul(np.linalg.inv(D), b )

    eig,_ = np.linalg.eig(T)
    spectral_radius = np.abs(eig).max()


    # print(f"T : {T}")
    # print(f"C : {C}")
    start_time = time.time()                                                                                                                                                                
    while (rel_diff > tol) and(iteration < maxiter):

        iteration+=1

        x = np.matmul(T,x_prev) + C
        # print(f"X {x}")
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

# A = np.array([[6.0,3.0],[3.0,9.0]])
# b = np.array([9.0,12.0])

# # print(A)

# # guess = array([0.0,0.0])

# x,iteration,total_time, all_x = jacobi(A,b,None,10 ** (-15), 250)
# print(all_x)
# plot_x(all_x)

# print( x, iteration, total_time )

# print(sol)
# A = array([[6.0,3.0],[3.0,9.0]])
# b = array([9.0,12.0])
# print(jacobi2(A,b,guess,1E-6))



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
                A[i][i] = np.random.randint(10 * dim,20 * dim)
            else :
                A[i][i] = -(np.random.randint(10 * dim,20 * dim))
        I = np.linalg.inv(A)
        b = np.random.randint(-1000,1000, size=dim)
        X = np.matmul(I,b)
    print(A)
    print(b)
    x,iteration,total_time, all_x, spectral_radius = jacobi(A,b,X,1E-15,200)
    # print('\nIterations=', iteration, 'Time=', time_taken)
    # print(f"Seed = {seed}")