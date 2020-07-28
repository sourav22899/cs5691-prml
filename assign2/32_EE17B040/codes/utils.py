import numpy as np

def linear_kernel(X,Y):
    assert X.shape[1] == Y.shape[1]
    K = np.matmul(X,Y.T)
    return K

def polynomial_kernel(X,Y,degree=2,gamma=None,a=1):
    assert X.shape[1] == Y.shape[1]
    if gamma is None:
        gamma = 1./X.shape[1]
    K = np.matmul(X,Y.T)
    K *= gamma
    K += a
    K **= degree
    return K