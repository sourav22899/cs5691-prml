"""# 1.1 Kernel Ridge Regression"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from utils import *

data = pd.read_csv('Regression_dataset.csv')
y = data['Y']
data = data.drop(columns='Y')

X = data.loc[:,]

X = np.asarray(X,dtype=np.float)
y = np.asarray(y,dtype=np.float)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print(X_train.shape)
print(X_test.shape)


def KernelRidgeRegression(X_train,y_train,X_test,y_test,\
                   kernel='linear',\
                   degree=3,\
                   gamma=None,\
                   a=1,\
                   lambda_=1):
    if kernel == 'linear':
        K = linear_kernel(X_train,X_train)
    elif kernel == 'poly':
        K = polynomial_kernel(X_train,X_train,degree=degree,gamma=gamma,a=a)
    else:
        print('Invalid Kernel !')
        return
    alpha = np.matmul(np.linalg.inv(K+lambda_*np.eye(K.shape[0])),y_train)
    if kernel == 'linear':
        h = linear_kernel(X_train,X_test)
    elif kernel == 'poly':
        h = polynomial_kernel(X_train,X_test,degree=degree,gamma=gamma,a=a)

    y_preds = np.matmul(alpha.T,h)
    y_train_preds = np.matmul(alpha.T,K)
    return y_preds, np.linalg.norm(y_test-y_preds)**2/y_test.shape[0],np.linalg.norm(y_train-y_train_preds)**2/y_train.shape[0]

lambda_list = np.logspace(-4,4,9)
for lambda_ in lambda_list:
    y_pred, te_error, tr_error = KernelRidgeRegression(X_train,y_train,X_test,y_test,kernel='linear',lambda_=lambda_)
    print('lambda: {}, Test Error:{:.6f}, Train Error:{:.6f}'.format(lambda_,te_error, tr_error))

lambda_list = np.logspace(-4,4,9)
degree_list = np.arange(5)
for degree in degree_list:
    for lambda_ in lambda_list:
        y_pred, te_error, tr_error = KernelRidgeRegression(X_train,y_train,X_test,y_test,\
                                                    kernel='poly',lambda_=lambda_,degree=degree)
        print('degree: {}, lambda: {}, Test Error:{:.6f}, Train Error:{:.6f}'.format(degree,lambda_,te_error, tr_error))
    print('-'*50)
