

# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 19:32:05 2019

@author: Karra's
"""

import numpy as np
from scipy.linalg import solve
from numpy import linalg as LA
import timeit
import matplotlib.pyplot as plt
from numpy import random



class Algorithms():
    
    def __init__(self,matrixsize):
        self.matrixsize=matrixsize
        self.x_initial=[1]*matrixsize
        self.b = random.rand(matrixsize)       
        self.p = random.rand(matrixsize,matrixsize)    
        self.A=np.add(0.5 *np.add(self.p,self.p.transpose()), matrixsize*np.identity(matrixsize))
    
    def gauss(self,n,max_error):
        L = np.tril(self.A)
        U = self.A - L
        x=self.x_initial
        x_prev=x[:]
        error_list=[]
        i=0
        error=float('inf')
       # D = np.diag(np.diag(A))
        #R = A -D
    
        T = np.linalg.solve(-L,U)
        eigsT, _ = np.linalg.eig(T)
        spectral_radius = np.max(np.abs(eigsT))
       
        start = timeit.default_timer()
     
        while i<n  :
            x = np.dot(np.linalg.inv(L), self.b - np.dot(U, x))
            error=LA.norm(x-x_prev)
            
            error_list.append(error)
            x_prev=x[:]
            if error<=max_error:
                break
            print (str(i).zfill(3))
            print(x)
            print(error)
        
            i=i+1
        stop = timeit.default_timer()
        print('Time: ', stop - start)
        return x,error_list,i,spectral_radius,stop - start 

    def jacobi(self,n,max_error):

        #L = np.tril(A)
        #U = A - L
        x=self.x_initial
        x_prev=x[:]
        error_list=[]
        i=0
        error=float('inf')
       # D = np.diag(np.diag(A))
        #R = A -D
        
        D = np.diag(self.A)
        R = self.A - np.diagflat(D)
        D_=np.diag(np.diag(self.A))
        T = np.linalg.solve(-D_,R)
        eigsT, _ = np.linalg.eig(T)
        spectral_radius = np.max(np.abs(eigsT))
       
        start = timeit.default_timer()
     
        while i<n  :
            x = (self.b - np.dot(R,x))/ D
            error=LA.norm(x-x_prev)
            
            error_list.append(error)
            x_prev=x[:]
            print(error,max_error)
            if error<=max_error:
                break
            print (str(i).zfill(3))
            print(x)
            print(error)
        
            i=i+1
        stop = timeit.default_timer()
        print('Time: ', stop - start)
        return x,error_list,i,spectral_radius,stop - start 


    def SOR(self,n,max_error,w):
        x=self.x_initial
        D_L = np.tril(self.A)
        D = np.diag(np.diag(self.A))
        L=D_L-D
        U = self.A-L-D
        x_prev=x[:]
        error_list=[]
        i=0
        error=float('inf')
       # D = np.diag(np.diag(A))
        #R = A -D
    
        T = np.linalg.solve(-(D+(w*L)),((1-w)*L)+U)
        eigsT, _ = np.linalg.eig(T)
        spectral_radius = np.max(np.abs(eigsT))
       
        start = timeit.default_timer()
     
        while i<n  :
            x = np.dot(np.linalg.inv(D+(w*L)), self.b - np.dot(U+(1-w)*L, x))
            error=LA.norm(x-x_prev)
            
            error_list.append(error)
            x_prev=x[:]
            if error<=max_error:
                break
            print (str(i).zfill(3))
            print(x)
            print(error)
        
            i=i+1
        stop = timeit.default_timer()
        print('Time: ', stop - start)
        return x,error_list,i,spectral_radius,stop - start 

    def plots(self,algo):
        
        max_iteration_list=[]
        size_list=[]
        time_list=[]
        spec_rad_list=[]
        max_iteration_list_g,max_iteration_list_j,max_iteration_list_s=[],[],[]
        size_list_g,size_list_j,size_list_s=[],[],[]
        time_list_g,time_list_j,time_list_s=[],[],[]
        spec_rad_list_g,spec_rad_list_j,spec_rad_list_s=[],[],[]              
        for i in range(10,100,10):
            x=Algorithms(i)
            if algo=='gauss':
                x_,error_l,max_iteration,spec_rad,time = x.gauss(float('inf'),1E-15)
            elif algo=='jacobi':
                x_,error_l,max_iteration,spec_rad,time = x.jacobi(float('inf'),1E-15)
            elif algo=='sor':
                x_,error_l,max_iteration,spec_rad,time = x.SOR(float('inf'),1E-15,0)
            elif algo=='all':
                x_g,error_l_g,max_iteration_g,spec_rad_g,time_g = x.gauss(float('inf'),1E-15)
                x_j,error_l_j,max_iteration_j,spec_rad_j,time_j = x.jacobi(float('inf'),1E-15)
                x_s,error_l_s,max_iteration_s,spec_rad_s,time_s = x.SOR(float('inf'),1E-15,0)
                size = size_list[:]
                

                spec_rad_list_g.append(spec_rad_g)
                size_list_g.append(i)
                time_list_g.append(time_g)
                max_iteration_list_g.append(max_iteration_g)
                
                spec_rad_list_j.append(spec_rad_j)
                size_list_j.append(i)
                time_list_j.append(time_j)
                max_iteration_list_j.append(max_iteration_j)

                spec_rad_list_s.append(spec_rad_s)
                size_list_s.append(i)
                time_list_s.append(time_s)
                max_iteration_list_s.append(max_iteration_s)
            if algo!="all":                 
                spec_rad_list.append(spec_rad)
                size_list.append(i)
                time_list.append(time)
                max_iteration_list.append(max_iteration)
                
        if algo=='all':
                size=size_list_g
                fig = plt.figure(figsize=(9,9))
                ax1 = fig.add_subplot(311) 
                ax2 = fig.add_subplot(312)
                ax3 = fig.add_subplot(313)
                ax1.set_title('size vs iterations')
                ax2.set_title('size vs time')
                ax3.set_title('size vs spectral_radius')
                ax1.plot(size, max_iteration_list_g,label = 'Gauss')
                ax1.plot(size, max_iteration_list_j,label = 'Jacobi')
                ax1.plot(size, max_iteration_list_s,label = 'SOR')
                ax1.legend()
                ax2.plot(size, time_list_g,label = 'Gauss')
                ax2.plot(size, time_list_j,label = 'Jacobi')
                ax2.plot(size, time_list_s,label = 'SOR')
                ax2.legend()
                ax3.plot(size, spec_rad_list_g,label = 'Gauss')
                ax3.plot(size, spec_rad_list_j,label = 'Jacobi')
                ax3.plot(size, spec_rad_list_s,label = 'SOR')
                ax3.legend()
                print(spec_rad_list_g,spec_rad_list_j,spec_rad_list_s)
                
                ax1.grid(True)
                ax2.grid(True)
                ax3.grid(True)
                
                
        else:

            size = size_list[:]
            
            fig = plt.figure(figsize=(9,9))
            ax1 = fig.add_subplot(311) 
            ax2 = fig.add_subplot(312)
            ax3 = fig.add_subplot(313)
            ax1.set_title('size vs iterations')
            ax2.set_title('size vs time')
            ax3.set_title('size vs spectral_radius')
            ax1.plot(size, max_iteration_list)
            ax2.plot(size, time_list)
            ax3.plot(size, spec_rad_list)
            
            ax1.grid(True)
            ax2.grid(True)
            ax3.grid(True)

            
    def SOR_plot(self,msize):

        spec_rad_list=[] 

        def spectral_radius(size,w):
            x=Algorithms(size)
            
            A_=x.A
            D_L = np.tril(A_)
            D = np.diag(np.diag(A_))
            L=D_L-D
            U = A_-L-D

            T = np.linalg.solve(-(D+(w*L)),((1-w)*L)+U)
            eigsT, _ = np.linalg.eig(T)
            spectral_rad = np.max(np.abs(eigsT))
            return spectral_rad
            
        for w in np.arange(0,2,0.2):
            spec_rad_list.append(spectral_radius(msize,w))
            
            
        fig,ax = plt.subplots()
        
        ax.set_title('w vs spec_radius')
        ax.plot([ p for p in np.arange(0,2,0.2)], spec_rad_list)            
        ax.grid(True)
    
    def rate_plots(self,d):
        x=Algorithms(d)
        w=0.5
        x_g,error_l_g,max_iteration_g,spec_rad_g,time_g = x.gauss(float('inf'),1E-15)
        x_j,error_l_j,max_iteration_j,spec_rad_j,time_j = x.jacobi(float('inf'),1E-15)
        x_s,error_l_s,max_iteration_s,spec_rad_s,time_s = x.SOR(float('inf'),1E-15,w)

        fig,ax = plt.subplots()
        
        ax.set_title('Error vs Iterations')
        m=min(len(error_l_g),len(error_l_j),len(error_l_s))
        x_axis_coord=[p for p in range(m)]
        ax.plot(x_axis_coord, error_l_g[:m],label = 'Gauss')
        ax.plot(x_axis_coord, error_l_j[:m],label = 'Jacobi')
        ax.plot(x_axis_coord, error_l_s[:m],label = 'SOR ('+str(w)+')')    
        ax.legend()        
        ax.grid(True)
        

               
if __name__=='__main__':
    alg=Algorithms(10)
    
    max_iterations_=float('inf')
    max_error=1E-15
    w=0.01
    msize=10
    
    ## uncomment each line and run individually
    #alg.gauss(max_iterations_,max_error)
   # alg.jacobi(max_iterations_,max_error)
   # alg.SOR(max_iterations_,max_error,w)
   # alg.plots("gauss")
   # alg.plots("jacobi")
   # alg.plots("sor")
   # alg.plots("all")
   # alg.SOR_plot(msize)
    alg.rate_plots(10)
            
