import numpy as np
import matplotlib.pyplot as plt

"""# **Question 6**"""

X = np.random.rand(100)
y = np.exp(np.sin(2*np.pi*X)) + np.random.normal(loc=0.0,scale=np.sqrt(0.2),size=X.shape[0])
plt.grid()
plt.scatter(X,y)
plt.show()

split = np.random.choice(X.shape[0],size=int(0.8*X.shape[0]),replace=False)
split_ = np.asarray([i for i in np.arange(X.shape[0]) if i not in split])
train_data_x = X[split]
train_data_y = y[split]
test_data_x = X[split_]
test_data_y = y[split_]

train_data = np.asarray(list((zip(train_data_x,train_data_y))))
test_data = np.asarray(list((zip(test_data_x,test_data_y))))
plt.scatter(train_data[:,0],train_data[:,1])
plt.show()

"""## **Question 6a**"""

idx = np.random.choice(train_data.shape[0],10,replace=False)
data = train_data[idx]
plt.scatter(data[:,0],data[:,1])
plt.show()

def poly_regression(data,deg=1):
  A = np.zeros((data.shape[0],deg+1))
  b = data[:,1]
  for i,x in enumerate(data[:,0]):
    A[i] = np.asarray([x**i for i in range(A.shape[1])])

  soln = np.linalg.lstsq(A,b,rcond=None)[0]
  y_hat = np.polyval(np.flip(soln),data[:,0])
  t_error = rmse(data[:,1],y_hat)
  return soln, t_error

def plot_poly_regression(data=data,soln=None):
  deg = soln.shape[0]-1
  x_ = np.linspace(0,1,101)
  y_hat = np.polyval(np.flip(soln),x_)

  plt.figure(figsize=(9,6))
  plt.title(r'no.of datapoints = {}, degree = {}'.format(data.shape[0],deg),size=15)
  plt.grid()
  plt.plot(x_,y_hat,color='red')
  plt.scatter(data[:,0],data[:,1])
  plt.xlabel(xlabel=r'x$\rightarrow$',size=15)
  plt.ylabel(ylabel=r'$\hat{y}\rightarrow$',size=15)
  plt.show()

def rmse(a,b):
  return np.sqrt(np.mean((a-b)**2))

sol1,_ = poly_regression(data,deg=1)
sol3,_ = poly_regression(data,deg=3)
sol6,_ = poly_regression(data,deg=6)
sol9,_ = poly_regression(data,deg=9)

plot_poly_regression(soln=sol1)
plot_poly_regression(soln=sol3)
plot_poly_regression(soln=sol6)
plot_poly_regression(soln=sol9)

"""## **Question 6b**"""

def overfit_test(test_data=test_data,train_data=train_data,deg=1):
  for n in [10,20,40,80]:
    idx = np.random.choice(train_data.shape[0],n,replace=False)
    data = train_data[idx]
    soln,train_error = poly_regression(data,deg=deg)
    y_hat_test = np.polyval(np.flip(soln),test_data[:,0])
    plt.figure(figsize=(9,6))
    plt.grid()
    plt.scatter(test_data[:,0],y_hat_test,label='predicted')
    plt.scatter(test_data[:,0],test_data[:,1],label='target')
    test_error = rmse(test_data[:,1],y_hat_test)
    plt.legend()
    print('n = {}, deg = {}, train_error = {}, test_error = {}'.format(n,deg,train_error,test_error))
    plt.show()

overfit_test(deg=6)

"""## **Question 6c**"""

def plot_target_output(test_data=test_data,train_data=train_data,deg=1):
    data = train_data
    soln,train_error = poly_regression(data,deg=deg)
    y_hat_test = np.polyval(np.flip(soln),test_data[:,0])
    plt.grid()
    plt.scatter(test_data[:,0],y_hat_test,label='predicted')
    plt.scatter(test_data[:,0],test_data[:,1],label='target')
    test_error = rmse(test_data[:,1],y_hat_test)
    plt.legend()
    print('n = {}, deg = {}, train_error = {}, test_error = {}'.format(data.shape[0],deg,train_error,test_error))
    plt.show()

plot_target_output(deg=9)

"""Degree 3 polynomial has the best generalization.

## **Question 6d**
"""

def find_rmse(train_data=train_data,test_data=test_data):
  train_errors,test_errors = [],[]
  for i in range(1,10):
    soln,train_error = poly_regression(data,deg=i)
    y_hat_test = np.polyval(np.flip(soln),test_data[:,0])
    test_error = rmse(test_data[:,1],y_hat_test)
    train_errors.append(train_error)
    test_errors.append(test_error)
  
  return train_errors, test_errors

train_errors, test_errors = find_rmse()

plt.figure(figsize=(9,6))
plt.title(r'RMSE with degree of polynomial',size=15)
plt.grid()
plt.plot(np.arange(1,10),train_errors,label='train_error')
plt.plot(np.arange(1,10),test_errors,label='test_error')
plt.xlabel(xlabel=r'degree of polynomial$\rightarrow$',size=15)
plt.ylabel(ylabel=r'RMSE$\rightarrow$',size=15)
plt.legend()
plt.show()