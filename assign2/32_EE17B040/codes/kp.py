import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from utils import *

data = pd.read_csv('Dataset_1_Team_32.csv')

plt.scatter(data['# x_1'],data['x_2'])
plt.show()

data['Class_label'] = np.sign(data['Class_label']-0.5)
y = data['Class_label']
X = data.drop(columns='Class_label')

X = np.asarray(X,dtype=np.float)
y = np.asarray(y,dtype=np.float)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print(X_train.shape,y_train.shape)
print(X_test.shape,y_test.shape)

def KernelPerceptron(X_train,y_train,X_test,y_test,\
                     max_iter=10,\
                     kernel='linear',\
                     degree=2,\
                     gamma=None,\
                     a=1):
    T = X_train.shape[0]
    alpha = np.zeros(T)
    if kernel == 'linear':
        for it in range(max_iter):
            up = 0
            for t in range(T):
                X_t = np.reshape(X_train[t],(1,2))
                h = linear_kernel(X_train,X_t)
                y_t_hat = np.sign(np.matmul((alpha*y_train).T,h))
                if y_train[t] != y_t_hat:
                    alpha[t] += 1
                    up += 1
            print('iter:{},updates:{}'.format(it+1,up))
        h = linear_kernel(X_train,X_test)
        y_test_preds = np.sign(np.matmul((alpha*y_train).T,h))
        h = linear_kernel(X_train,X_train)
        y_train_preds = np.sign(np.matmul((alpha*y_train).T,h))
    elif kernel == 'poly':
        for it in range(max_iter):
            up = 0
            for t in range(T):
                X_t = np.reshape(X_train[t],(1,2))
                h = polynomial_kernel(X_train,X_t,degree=degree,gamma=gamma,a=a)
                y_t_hat = np.sign(np.matmul((alpha*y_train).T,h))
                if y_train[t] != y_t_hat:
                    alpha[t] += 1
                    up += 1
            print('iter:{},updates:{}'.format(it+1,up))
        h = polynomial_kernel(X_train,X_test,degree=degree,gamma=gamma,a=a)
        y_test_preds = np.sign(np.matmul((alpha*y_train).T,h))
        h = polynomial_kernel(X_train,X_train,degree=degree,gamma=gamma,a=a)
        y_train_preds = np.sign(np.matmul((alpha*y_train).T,h))
    else:
        print('Invalid Kernel !')
        return
        
    return alpha, np.mean(y_test == y_test_preds), np.mean(y_train == y_train_preds), [degree, gamma, a]

def predict_poly(alpha,X,Y,y,degree=2,gamma=None,a=1):
    h = polynomial_kernel(X,Y,degree=degree,gamma=gamma,a=a)
    return np.sign(np.matmul((alpha*y).T,h))

def predict_linear(alpha,X,Y,y):
    h = linear_kernel(X,Y)
    return np.sign(np.matmul((alpha*y).T,h))

def predict(alpha,X,Y,y,kernel='linear',degree=2,gamma=None,a=1):
    if kernel == 'linear':
        return predict_linear(alpha,X,Y,y)
    elif kernel == 'poly':
        return predict_poly(alpha,X,Y,y,degree=degree,gamma=gamma,a=a)
    else:
        print('Invalid Kernel !')
        return

kernel = 'linear'
alpha,te_acc,tr_acc,params = KernelPerceptron(X_train,y_train,X_test,y_test,max_iter=50,kernel=kernel,degree=3)
print('test accuracy: {}, train accuracy: {}'.format(te_acc,tr_acc))

# Finding margin
R = np.linalg.norm(X_train,axis=1).max()
clf = SVC(kernel='linear',C=1000,degree=3)
clf.fit(X_train, y_train)
print(clf.score(X_train, y_train))
print(clf.score(X_test, y_test))

w = clf.coef_
w1 = w/np.linalg.norm(w)
gamma = (y_train*np.matmul(w1,X_train.T)).min()

# Plot decision boundaries

dataset = 1
def make_meshgrid(x, y, dataset=3):
    h = 0.02 if dataset == 3 else 0.75
    x_min, x_max = x.min() - 25*h, x.max() + 25*h
    y_min, y_max = y.min() - 25*h, y.max() + 25*h
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy

def plot_contours(ax,\
                  alpha_,\
                  X_train,\
                  y,\
                  xx,\
                  yy,\
                  kernel='linear',\
                  degree=2,\
                  gamma=None,\
                  a=1,\
                  **params):
    X = np.c_[xx.ravel(), yy.ravel()]
    Z = predict(alpha_,X_train,X,y,kernel=kernel,degree=degree,gamma=gamma,a=a)
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

fig, ax = plt.subplots(figsize=(9,6))
# Set-up grid for plotting.

X0, X1 = X_train[:, 0], X_train[:, 1]
xx, yy = make_meshgrid(X0, X1,dataset=dataset)
degree, gamma, a = params
plot_contours(ax,\
              alpha_=alpha,\
              X_train=X_train,
              y=y_train,\
              xx=xx,\
              yy=yy,\
              kernel=kernel,\
              degree=degree,\
              gamma=gamma,\
              a=a,\
              cmap=plt.cm.autumn,\
              alpha=1)
ax.scatter(X0, X1, c=y_train, cmap=plt.cm.autumn, s=20, edgecolors='k',marker = 's')
ax.set_ylabel(r'$x_2\rightarrow$')
ax.set_xlabel(r'$x_1\rightarrow$')
ax.set_title('Decison surface for polynomial kernel perceptron for dataset 1')
plt.show()