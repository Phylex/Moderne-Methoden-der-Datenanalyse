#!/usr/bin/python3
# coding: utf-8
'''An example of an analytical linear fit to 5 datapoints
.. author: Alexander Becker
'''
import numpy as np
import matplotlib.pyplot as plt

X = np.array([1, 2, 3, 4, 5])
Y = np.array([1.94759, 1.90523, 2.65621, 4.20916, 3.44776])
Xuc = np.array([0., 0., 0., 0., 0.])
Yuc = np.array([0.10, 0.30, 0.25, 0.50, 0.75])

def gauss(domain, mean, stddev):
    '''a gaus distribution with parametrizable mean and stddev'''
    return 1/(np.sqrt(2*np.pi)*stddev) * np.exp((domain-mean)**2/stddev**2)

def true_function(domain):
    '''this function represents the underlying truth that the vals in Y(X) are sampled from'''
    return 0.6*domain+1.

def fit_function(domain, slope, intersect):
    '''use a first order polynomial to fit the data'''
    return slope*domain+intersect

def analytical_LS_fit(A, V, Y):
    '''an analytical least square fit using matricies to calculate it'''
    AT = np.transpose(A)
    VI = np.linalg.inv(V)
    return np.linalg.inv(AT @ VI @ A) @ AT @ VI @ Y

def analytical_LS_cov(A, V):
    '''calculate the covariance using the matrix multiplication method'''
    AT = np.transpose(A)
    VI = np.linalg.inv(V)
    return np.linalg.inv(AT @ VI @ A)

def analytical_chi2(A, V, Y, T):
    '''calculate the chi2 distance using the matrix method'''
    VI = np.linalg.inv(V)
    return np.transpose(Y - A @ T) @ VI @ (Y - A @ T)

def matprint(A, w=None, h=None):
    if A.ndim==1:
        if w == None :
            return str(A)
        else:
            s ='['+' '*(max(w[-1],len(str(A[0])))-len(str(A[0]))) +str(A[0])
            for i,AA in enumerate(A[1:]):
                s += ' '*(max(w[i],len(str(AA)))-len(str(AA))+1)+str(AA)
            s +='] '
    elif A.ndim==2:
        w1 = [max([len(str(s)) for s in A[:,i]])  for i in range(A.shape[1])]
        w0 = sum(w1)+len(w1)+1
        s= u'\u250c'+u'\u2500'*w0+u'\u2510' +'\n'
        for AA in A:
            s += ' ' + matprint(AA, w=w1) +'\n'
        s += u'\u2514'+u'\u2500'*w0+u'\u2518'
    elif A.ndim==3:
        h=A.shape[1]
        s1=u'\u250c' +'\n' + (u'\u2502'+'\n')*h + u'\u2514'+'\n'
        s2=u'\u2510' +'\n' + (u'\u2502'+'\n')*h + u'\u2518'+'\n'
        strings=[matprint(a)+'\n' for a in A]
        strings.append(s2)
        strings.insert(0,s1)
        s='\n'.join(''.join(pair) for pair in zip(*map(str.splitlines, strings)))
    return s
