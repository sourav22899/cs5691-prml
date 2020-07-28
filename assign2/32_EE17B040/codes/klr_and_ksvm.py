import numpy as rnp
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import jax.numpy as np
from jax import grad, jit

from utils import *

""" Kernel SVM """

data = pd.read_csv('Dataset_3_Team_32.csv')

y = data['Class_label']
X = data.drop(columns='Class_label')

X = rnp.asarray(X,dtype=rnp.float)
y = rnp.asarray(y,dtype=rnp.float)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print(X_train.shape)
print(X_test.shape)

clf = SVC(kernel='linear')
clf.fit(X_train, y_train)
print(clf.score(X_train,y_train))
print(clf.score(X_test,y_test))

for i in range(10):
    clf = SVC(gamma='auto',kernel='poly',degree=i)
    clf.fit(X_train, y_train)
    print(i, clf.score(X_train,y_train), clf.score(X_test,y_test), clf.n_support_)

#######################################################################################################

"""# Kernel Logistic Regression"""
y_train = np.sign(y_train-0.5)

def sigmoid(x):
    return 1./(1+np.exp(-x))

def update_alpha(alpha,K,y):
    delta_alpha = np.zeros(K.shape[0])
    for i in range(K.shape[0]):
        delta_alpha += -y[i]*K[:,i]*sigmoid(-y[i]*np.matmul(alpha.T,K[:,i]))
    return delta_alpha

def loss_function(alpha,K,y,lambda_=1):
    J = np.log(1+np.exp(-y*np.matmul(alpha.T,K))).sum() + 0.5*lambda_*alpha.T.dot(K.dot(alpha))
    return J/K.shape[0]

grad_J = jit(grad(loss_function, argnums=0))

K = linear_kernel(X_train,X_train)
alpha = np.zeros(X_train.shape[0])
lr = 0.01
max_iters = 1000
for i in range(max_iters):
    J = loss_function(alpha,K,y_train)
    d_alpha = grad_J(alpha,K,y_train)
    alpha = alpha - lr*(0.9999)**i*d_alpha
    if (i+1) % 100 == 0:
      print('iters:{}, grad_norm:{}, loss:{}'.format(i+1,np.linalg.norm(d_alpha),J))

h = linear_kernel(X_train,X_test)
Z = np.matmul(alpha.T,h)

y_preds = np.asarray(sigmoid(Z) > 0.5,dtype=np.int32)

y_train = (y_train+1)*0.5
print(np.mean(y_preds == y_test))

import numpy as np

dataset = 3 # Choose the dataset
def make_meshgrid(x, y, dataset=3):
    h = 0.02 if dataset == 3 else 0.75
    x_min, x_max = x.min() - 25*h, x.max() + 25*h
    y_min, y_max = y.min() - 25*h, y.max() + 25*h
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy

# Set-up grid for plotting.
X0, X1 = X_train[:, 0], X_train[:, 1]
xx, yy = make_meshgrid(X0, X1,dataset=dataset)
X = np.c_[xx.ravel(), yy.ravel()]

h = linear_kernel(X_train,X) # Choose the kernel
Z = np.matmul(alpha.T,h)
y_preds = np.asarray(sigmoid(Z) > 0.5,dtype=np.int32)
y_preds_f = y_preds.reshape(xx.shape)
plt.figure(figsize=(9,6))
plt.contourf(xx,yy,y_preds_f,cmap=plt.cm.autumn)
plt.scatter(X0, X1, c=y_train, cmap=plt.cm.autumn, s=20, edgecolors='k',marker = 's')
plt.ylabel(r'$x_2\rightarrow$',size=15)
plt.xlabel(r'$x_1\rightarrow$',size=15)
plt.title('Decision surface and data points for Dataset 3 for linear kernel LR',size=15)
plt.show()