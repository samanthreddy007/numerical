from jacobi import jacobi
from gauss import gauss
from sor import SOR,optimal_w
from TriangleAlgorithm import TriangleAlgo
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # X = [-1,0,0,0]
    # y = 0
    
    ''' Find valid A and b '''
    
    # # while(not(all(x >= 0 for x in X))):
    # y = y+1
    # #print(y)
    # dim = 20
    # a = np.random.rand(dim,dim)
    # A = (a + a.T)/2
    # A = np.add(dim * np.identity(dim), A)
    # # for i in range(dim):
    # #     if((y+i)%3 == 0):
    # #         A[i][i] = np.sum(np.absolute(A[i])) + 5

    # #     else :
    # #         #A[i][i] = -(np.random.randint(10 * dim,20 * dim))
    # #         A[i][i] = -(np.sum(np.absolute(A[i]))+ 5)

    # # I = np.linalg.inv(A)
    # b = np.random.rand(dim)
    # # X = np.matmul(I,b)
    # print(A)
    # print(b)



    # A = np.array([[ 7,-2,1,2], [2,8,3,1], [-1,0,5,2], [0,2,-1,4] ])
    # b = np.array([3,-2,5,4])
    # x_tri, iteration_tri, seed_tri, time_taken_tri = TriangleAlgo(A,b,error = 0.1)
    # print("Running Gaussian")
    # x_gauss,iteration_gauss,total_time_gauss, all_x_gauss, spectral_radius_gauss = gauss(A,b,None,1E-15,1000000)
    # print("Running Jacobi")
    # x_jacobi,iteration_jacobi,total_time_jacobi, all_x_jacobi, spectral_radius_jacobi = jacobi(A,b,None,1E-15,1000000)
    # print("Running Sor")
    # x_sor,iteration_sor,total_time_sor, all_x_sor, spectral_radius_sor = SOR(A,b,None,1E-15,1000000,optimal_w(A))
    # print('\nIterations=', iteration, 'Time=', time_taken)
    # print(f"Seed = {seed}")

    iteration_d = {
        "gauss":[],
        "jacobi":[],
        "sor":[]
    }

    time_d = {
        "gauss":[],
        "jacobi":[],
        "sor":[]
    }

    spectral_d = {
        "gauss":[],
        "jacobi":[],
        "sor":[]
    }


    for dim in range(10,100,5) :

        a = np.random.rand(dim,dim)
        A = (a + a.T)/2
        A = np.add(dim * np.identity(dim), A)

        b = np.random.rand(dim)
        
        
        print("Running Gaussian")
        x_gauss,iteration_gauss,total_time_gauss, all_x_gauss, spectral_radius_gauss = gauss(A,b,None,1E-15,1000000)
        print("Running Jacobi")
        x_jacobi,iteration_jacobi,total_time_jacobi, all_x_jacobi, spectral_radius_jacobi = jacobi(A,b,None,1E-15,1000000)
        print("Running Sor")
        x_sor,iteration_sor,total_time_sor, all_x_sor, spectral_radius_sor = SOR(A,b,None,1E-15,1000000,optimal_w(A))

        iteration_d["gauss"].append(iteration_gauss)
        iteration_d["jacobi"].append(iteration_jacobi)
        iteration_d["sor"].append(iteration_sor)

        time_d["gauss"].append(total_time_gauss)
        time_d["jacobi"].append(total_time_jacobi)
        time_d["sor"].append(total_time_sor)

        spectral_d["gauss"].append(spectral_radius_gauss)
        spectral_d["jacobi"].append(spectral_radius_jacobi)
        spectral_d["sor"].append(spectral_radius_sor)


    plt.figure(figsize=(10,5))
    plt.title("Iteration vs Size")
    plt.plot(np.arange(10,100,5),iteration_d["gauss"],label="Gauss")
    plt.plot(np.arange(10,100,5),iteration_d["jacobi"],label="Jacobi")
    plt.plot(np.arange(10,100,5),iteration_d["sor"],label="SOR")
    plt.legend()
    plt.savefig("iteration.jpg")

    plt.figure(figsize=(10,5))
    plt.title("Time vs Size")
    plt.plot(np.arange(10,100,5),time_d["gauss"],label="Gauss")
    plt.plot(np.arange(10,100,5),time_d["jacobi"],label="Jacobi")
    plt.plot(np.arange(10,100,5),time_d["sor"],label="SOR")
    plt.legend()
    plt.savefig("time.jpg")

    plt.figure(figsize=(10,5))
    plt.title("Spectral vs Size")
    plt.plot(np.arange(10,100,5),spectral_d["gauss"],label="Gauss")
    plt.plot(np.arange(10,100,5),spectral_d["jacobi"],label="Jacobi")
    plt.plot(np.arange(10,100,5),spectral_d["sor"],label="SOR")
    plt.legend()
    plt.savefig("spectral.jpg")