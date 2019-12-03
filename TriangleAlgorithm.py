# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 03:46:19 2019

@author: pc
"""

import numpy as np
import math
import time

def TriangleAlgo(A, b, error):
        print('Running Triangle Algorithm.......')
        vectors = []
        flag = True
        p = np.zeros(len(A))
        min_norm = np.linalg.norm(b)
        max_norm = 0
        min_index = -1
        p1 = -b
        alphas = np.zeros(len(A)+1)
        found_seed = False

        
        ''' Find the vertices of convex '''
        
        for index,col in enumerate(A.transpose()):
            vectors.append(col)
        
            if(np.linalg.norm(col)< min_norm):
                p1 = col
                min_norm = np.linalg.norm(col)
                min_index = index
            if(np.linalg.norm(col)> max_norm):
                max_norm = np.linalg.norm(col)  
        vectors.append(-b)
        vectors = np.array(vectors)
        alphas[min_index] = 1
        iteration = 0
        start = time.time()
        while(1):
            
            iteration = iteration + 1
            
            
            
            ''' Find Pivot '''
            for index, vector in enumerate(vectors):
                if(np.linalg.norm(p1 - vector) > np.linalg.norm(vector)):
                    vj = vector
                    j = index
                    break
    
        
            ''' Find P'' '''
            if((np.dot(p-p1, vj-p1)/np.linalg.norm(vj-p1)) > np.linalg.norm(vj-p1)):
                print('entered')
                p1 = vj
                alphas = np.zeros(len(A)+1)
                alphas[j] = 1
                flag = True
                
            else:
            
                alpha = np.dot(p-p1, vj-p1)/math.pow(np.linalg.norm(vj-p1),2)
                temp = alphas[j]
                alphas = (1 - alpha) * alphas
                alphas[j] = (1 - alpha) * temp + alpha
                '''
                new_p = np.zeros(len(alphas)-1)
            
                for i in range(len(alphas)):
                    
                    new_p = new_p + alphas[i] * vectors[i]'''
                p1 = (1 - alpha)* p1 + alpha * vj
                #p1 = new_p
                flag = False
                
                
            ''' Return x if close enough '''
            
            if(flag or alphas[-1] == 0):
                continue
            
           
            else:

                x = alphas[:(len(alphas)-1)]/alphas[-1]
                if(not(found_seed)):
                    seed = x
                    found_seed = True
                
                if(np.linalg.norm(np.matmul(A,x) - b) > error):
                    if(iteration % 1000 == 0):
                        print('Iteration=',iteration, 'error=', np.linalg.norm(np.matmul(A,x) - b))
                    continue
                else:
                    
                    return x, iteration,seed, time.time() - start





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
    x, iteration, seed, time_taken = TriangleAlgo(A,b,error = 0.1)
    print('\nIterations=', iteration, 'Time=', time_taken)