import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from collections import Counter

np.set_printoptions(suppress=True)

dataset_no = int(input('Dataset Number:'))
df = pd.read_csv('Dataset_'+str(dataset_no)+'_Team_37.csv')
train = df.sample(frac=0.8)
test = df.drop(train.index)
N = len(train)
M = len(test)
classes = [0,1,2]
c = len(classes)
d = 2

# normalizing

x_1_mean = train['# x_1'].mean()
x_1_std = train['# x_1'].std()
x_2_mean = train['x_2'].mean()
x_2_std = train['x_2'].std()

train['# x_1'] = (train['# x_1']-x_1_mean)/x_1_std
train['x_2'] = (train['x_2']-x_2_mean)/x_2_std
test['# x_1'] = (test['# x_1']-x_1_mean)/x_1_std
test['x_2'] = (test['x_2']-x_2_mean)/x_2_std

train_data_0 = train.loc[train['Class_label'] == 0]
train_data_1 = train.loc[train['Class_label'] == 1]
train_data_2 = train.loc[train['Class_label'] == 2]

prior_prob = Counter(train['Class_label'])
for key in prior_prob:
  prior_prob[key] = prior_prob[key]/float(N)

loss_matrix = np.asarray([[0,2,1],[2,0,3],[1,3,0]])

def calc_f_value(mu,var,x):
  C = (2*np.pi*np.sqrt(np.linalg.det(var)))**(-1) 
  return C*np.exp(-0.5*np.inner((x-mu),np.matmul(np.linalg.pinv(var),(x-mu))))

def bayes_classifier(params=None,x=None,loss_matrix=loss_matrix,prior_prob=prior_prob):
  # x is 2d vector
  mu = params['mu']
  var = params['var']
  q = np.zeros(c)
  for i in classes:  
    for j in classes:
      q[i] += loss_matrix[i,j]*calc_f_value(mu[j],var[j],x)*prior_prob[j]
    
  return np.argmin(q)

"""### **Question 1a**"""

print('*'*50 + 'Model_1'+ '*'*50)
mu = np.asarray([[np.mean(train_data_0['# x_1']),np.mean(train_data_0['x_2'])],
                [np.mean(train_data_1['# x_1']),np.mean(train_data_1['x_2'])],
                [np.mean(train_data_2['# x_1']),np.mean(train_data_2['x_2'])]])
var = np.asarray([np.eye(d),np.eye(d),np.eye(d)])

params = {}
params['mu'] = mu
params['var'] = var

train_data = np.array(train[['# x_1','x_2']].apply(list,axis=1))
train_data = np.reshape(train_data.tolist(),(N,2))
predicted = np.zeros(len(train))
for i,x in enumerate(train_data):
  predicted[i] = int(bayes_classifier(params,x))

train['predicted'] = predicted
print(np.mean(train['Class_label'] == train['predicted']))
print(confusion_matrix(np.asarray(train['Class_label']),np.asarray(train['predicted'])))

test_data = np.array(test[['# x_1','x_2']].apply(list,axis=1))
test_data = np.reshape(test_data.tolist(),(M,2))
predicted = np.zeros(len(test))

predicted_dict = {}
for i in classes:
  predicted_dict[i] = []
for i,x in enumerate(test_data):
  pred_class = int(bayes_classifier(params,x))
  predicted[i] = pred_class
  predicted_dict[pred_class].append(x)

for i in predicted_dict:
  predicted_dict[i] = np.reshape(np.asarray(predicted_dict[i]),(-1,2))
test['predicted'] = predicted
print(np.mean(test['Class_label'] == test['predicted']))
print(confusion_matrix(np.asarray(test['Class_label']),np.asarray(test['predicted'])))

# rescaling
def denormalize(x,std,mean):
  return std*x + mean

for i in predicted_dict:
  predicted_dict[i][:,0] = denormalize(predicted_dict[i][:,0],x_1_std,x_1_mean)
  predicted_dict[i][:,1] = denormalize(predicted_dict[i][:,1],x_2_std,x_2_mean)

"""### **Question 1b**"""

print('*'*50 + 'Model_2'+ '*'*50)
mu = np.asarray([[np.mean(train_data_0['# x_1']),np.mean(train_data_0['x_2'])],
                [np.mean(train_data_1['# x_1']),np.mean(train_data_1['x_2'])],
                [np.mean(train_data_2['# x_1']),np.mean(train_data_2['x_2'])]])
var = np.asarray([np.diag([np.var(train_data_0['# x_1']),np.var(train_data_0['x_2'])]),
                  np.diag([np.var(train_data_1['# x_1']),np.var(train_data_1['x_2'])]),
                  np.diag([np.var(train_data_2['# x_1']),np.var(train_data_2['x_2'])])])
var = np.mean(var,axis=0)

params = {}
params['mu'] = mu
params['var'] = np.array([var,var,var])

train_data = np.array(train[['# x_1','x_2']].apply(list,axis=1))
train_data = np.reshape(train_data.tolist(),(N,2))
predicted = np.zeros(len(train))
for i,x in enumerate(train_data):
  predicted[i] = int(bayes_classifier(params,x))

train['predicted'] = predicted
print(np.mean(train['Class_label'] == train['predicted']))
print(confusion_matrix(np.asarray(train['Class_label']),np.asarray(train['predicted'])))

test_data = np.array(test[['# x_1','x_2']].apply(list,axis=1))
test_data = np.reshape(test_data.tolist(),(M,2))
predicted = np.zeros(len(test))
for i,x in enumerate(test_data):
  predicted[i] = int(bayes_classifier(params,x))

test['predicted'] = predicted
print(np.mean(test['Class_label'] == test['predicted']))
print(confusion_matrix(np.asarray(test['Class_label']),np.asarray(test['predicted'])))

# test_data[:,0] = denormalize(test_data[:,0],x_1_std,x_1_mean)
# test_data[:,1] = denormalize(test_data[:,1],x_2_std,x_2_mean)
# params['mu'][:,0] = denormalize(params['mu'][:,0],x_1_std,x_1_mean)
# params['mu'][:,1] = denormalize(params['mu'][:,1],x_2_std,x_2_mean)

# plt.grid()
# plt.scatter(test_data[:,0],test_data[:,1])
# plt.scatter(params['mu'][:,0],params['mu'][:,1])
# plt.show()

"""### **Question 1c**"""
print('*'*50 + 'Model_3'+ '*'*50)
mu = np.asarray([[np.mean(train_data_0['# x_1']),np.mean(train_data_0['x_2'])],
                [np.mean(train_data_1['# x_1']),np.mean(train_data_1['x_2'])],
                [np.mean(train_data_2['# x_1']),np.mean(train_data_2['x_2'])]])
var = np.asarray([np.diag([np.var(train_data_0['# x_1']),np.var(train_data_0['x_2'])]),
                  np.diag([np.var(train_data_1['# x_1']),np.var(train_data_1['x_2'])]),
                  np.diag([np.var(train_data_2['# x_1']),np.var(train_data_2['x_2'])])])

params = {}
params['mu'] = mu
params['var'] = var

train_data = np.array(train[['# x_1','x_2']].apply(list,axis=1))
train_data = np.reshape(train_data.tolist(),(N,2))
predicted = np.zeros(len(train))
for i,x in enumerate(train_data):
  predicted[i] = int(bayes_classifier(params,x))

train['predicted'] = predicted
print(np.mean(train['Class_label'] == train['predicted']))
print(confusion_matrix(np.asarray(train['Class_label']),np.asarray(train['predicted'])))

test_data = np.array(test[['# x_1','x_2']].apply(list,axis=1))
test_data = np.reshape(test_data.tolist(),(M,2))

predicted = np.zeros(len(test))
for i,x in enumerate(test_data):
  predicted[i] = int(bayes_classifier(params,x))

test['predicted'] = predicted
print(np.mean(test['Class_label'] == test['predicted']))
print(confusion_matrix(np.asarray(test['Class_label']),np.asarray(test['predicted'])))

# test_data[:,0] = denormalize(test_data[:,0],x_1_std,x_1_mean)
# test_data[:,1] = denormalize(test_data[:,1],x_2_std,x_2_mean)
# params['mu'][:,0] = denormalize(params['mu'][:,0],x_1_std,x_1_mean)
# params['mu'][:,1] = denormalize(params['mu'][:,1],x_2_std,x_2_mean)

# plt.grid()
# plt.scatter(test_data[:,0],test_data[:,1])
# plt.scatter(params['mu'][:,0],params['mu'][:,1])
# plt.show()

"""### **Question 1d**"""
print('*'*50 + 'Model_4'+ '*'*50)
mu = np.zeros((c,d))
var = np.zeros((c,d,d))

train_data_0_ = np.array(train_data_0[['# x_1','x_2']].apply(list,axis=1))
train_data_0_ = np.reshape(train_data_0_.tolist(),(len(train_data_0),2))
mu[0] = np.mean(train_data_0_,axis=0)

train_data_0_ = train_data_0_ - mu[0]
for x in train_data_0_:
  var[0] += np.outer(x,x) 
var[0] = var[0]/train_data_0_.shape[0]

train_data_1_ = np.array(train_data_1[['# x_1','x_2']].apply(list,axis=1))
train_data_1_ = np.reshape(train_data_1_.tolist(),(len(train_data_1),2))
mu[1] = np.mean(train_data_1_,axis=0)

train_data_1_ = train_data_1_ - mu[1]
for x in train_data_1_:
  var[1] += np.outer(x,x)  
var[1] = var[1]/train_data_1_.shape[0]

train_data_2_ = np.array(train_data_2[['# x_1','x_2']].apply(list,axis=1))
train_data_2_ = np.reshape(train_data_2_.tolist(),(len(train_data_2),2))
mu[2] = np.mean(train_data_2_,axis=0)

train_data_2_ = train_data_2_ - mu[2]
for x in train_data_2_:
  var[2] += np.outer(x,x)  
var[2] = var[2]/train_data_2_.shape[0]

var = np.mean(var,axis=0)
params = {}
params['mu'] = mu
params['var'] = np.array([var,var,var])

train_data = np.array(train[['# x_1','x_2']].apply(list,axis=1))
train_data = np.reshape(train_data.tolist(),(N,2))

predicted = np.zeros(len(train))
for i,x in enumerate(train_data):
  predicted[i] = int(bayes_classifier(params,x))

train['predicted'] = predicted
print(np.mean(train['Class_label'] == train['predicted']))
print(confusion_matrix(np.asarray(train['Class_label']),np.asarray(train['predicted'])))

test_data = np.array(test[['# x_1','x_2']].apply(list,axis=1))
test_data = np.reshape(test_data.tolist(),(M,2))

predicted = np.zeros(len(test))
for i,x in enumerate(test_data):
  predicted[i] = int(bayes_classifier(params,x))

test['predicted'] = predicted
print(np.mean(test['Class_label'] == test['predicted']))
print(confusion_matrix(np.asarray(test['Class_label']),np.asarray(test['predicted'])))

"""### **Question 1e**"""
print('*'*50 + 'Model_5'+ '*'*50)
mu = np.zeros((c,d))
var = np.zeros((c,d,d))

train_data_0_ = np.array(train_data_0[['# x_1','x_2']].apply(list,axis=1))
train_data_0_ = np.reshape(train_data_0_.tolist(),(len(train_data_0),2))
mu[0] = np.mean(train_data_0_,axis=0)

train_data_0_ = train_data_0_ - mu[0]
for x in train_data_0_:
  var[0] += np.outer(x,x) 
var[0] = var[0]/train_data_0_.shape[0]

train_data_1_ = np.array(train_data_1[['# x_1','x_2']].apply(list,axis=1))
train_data_1_ = np.reshape(train_data_1_.tolist(),(len(train_data_1),2))
mu[1] = np.mean(train_data_1_,axis=0)

train_data_1_ = train_data_1_ - mu[1]
for x in train_data_1_:
  var[1] += np.outer(x,x)  
var[1] = var[1]/train_data_1_.shape[0]

train_data_2_ = np.array(train_data_2[['# x_1','x_2']].apply(list,axis=1))
train_data_2_ = np.reshape(train_data_2_.tolist(),(len(train_data_2),2))
mu[2] = np.mean(train_data_2_,axis=0)

train_data_2_ = train_data_2_ - mu[2]
for x in train_data_2_:
  var[2] += np.outer(x,x)  
var[2] = var[2]/train_data_2_.shape[0]

params = {}
params['mu'] = mu
params['var'] = var

train_data = np.array(train[['# x_1','x_2']].apply(list,axis=1))
train_data = np.reshape(train_data.tolist(),(N,2))

predicted = np.zeros(len(train))
for i,x in enumerate(train_data):
  predicted[i] = int(bayes_classifier(params,x))

train['predicted'] = predicted
print(np.mean(train['Class_label'] == train['predicted']))
print(confusion_matrix(np.asarray(train['Class_label']),np.asarray(train['predicted'])))

test_data = np.array(test[['# x_1','x_2']].apply(list,axis=1))
test_data = np.reshape(test_data.tolist(),(M,2))
predicted = np.zeros(len(test))
# for i,x in enumerate(test_data):
#   predicted[i] = int(bayes_classifier(params,x))

# test['predicted'] = predicted
# print(np.mean(test['Class_label'] == test['predicted']))
# print(confusion_matrix(np.asarray(test['Class_label']),np.asarray(test['predicted'])))

predicted_dict = {}
for i in classes:
  predicted_dict[i] = []
for i,x in enumerate(test_data):
  pred_class = int(bayes_classifier(params,x))
  predicted[i] = pred_class
  predicted_dict[pred_class].append(x)

for i in predicted_dict:
  predicted_dict[i] = np.reshape(np.asarray(predicted_dict[i]),(-1,2))
test['predicted'] = predicted
print(np.mean(test['Class_label'] == test['predicted']))
print(confusion_matrix(np.asarray(test['Class_label']),np.asarray(test['predicted'])))

# for i in predicted_dict:
#   predicted_dict[i][:,0] = denormalize(predicted_dict[i][:,0],x_1_std,x_1_mean)
#   predicted_dict[i][:,1] = denormalize(predicted_dict[i][:,1],x_2_std,x_2_mean)

def plot_scatter(data=predicted_dict,xlabel=None,ylabel=None,figsize=(9,6),model=1,dataset_no=1):
  plt.figure(figsize=figsize)
  plt.title(r'Scatter plot of model {} for dataset {}'.format(model,dataset_no),size=15)
  plt.grid()
  plt.scatter(data[0][:,0], data[0][:,1], marker='+', c='k')
  plt.scatter(data[1][:,0], data[1][:,1], marker='.', c='k')
  plt.scatter(data[2][:,0], data[2][:,1], marker='1', c='k')
  plt.xlabel(xlabel=r'input feature x1',size=15)
  plt.ylabel(ylabel=r'input feature x2',size=15)
  plt.show()

# plot_scatter(data=predicted_dict,model=1,dataset_no=1)

######################################################################################

from tqdm import tqdm
if dataset_no == 1:
  for i in predicted_dict:
    predicted_dict[i] = np.clip(predicted_dict[i], -1.9, 1.9)
else:
  for i in predicted_dict:
    predicted_dict[i] = np.clip(predicted_dict[i], -3.9, 3.9)

if dataset_no == 1:
  x = np.arange(-2.5,2.5,0.01)
  y = np.arange(-2.5,2.5,0.01)
  xticks = [str(i) for i in np.arange(-750,750,200)]
  yticks = [str(i) for i in np.arange(-150,250,50)]
else:
  x = np.arange(-4,4,0.02)
  y = np.arange(-4,4,0.02)
  xticks = [str(i) for i in np.arange(-100,100,25)]
  yticks = [str(i) for i in np.arange(-400,400,100)]
  
print(xticks,yticks)
def plot_decision_surface(data=predicted_dict,
                          params=params, 
                          xlabel=None, 
                          ylabel=None, 
                          figsize=(9,6),
                          dataset_no=1):
  if dataset_no == 1:
    x = np.arange(-2.5,2.5,0.02)
    y = np.arange(-2.5,2.5,0.02)
  else:
    x = np.arange(-4,4,0.02)
    y = np.arange(-4,4,0.02)
  z = np.zeros(x.shape[0]*y.shape[0])
  xx, yy = np.meshgrid(x,y)
  data_points = np.c_[xx.ravel(),yy.ravel()]
  for i,x in tqdm(enumerate(data_points)):
    z[i] = int(bayes_classifier(params,x))
  z = np.reshape(z,xx.shape)
  plt.title(r'Decision surface of best model for dataset {}'.format(dataset_no),size=15)
  plt.contourf(xx,yy,z)
  # plt.grid()
  plt.scatter(data[0][:,0], data[0][:,1], marker='^',facecolors='none', 
              edgecolors='r', label='Class 0')
  plt.scatter(data[1][:,0], data[1][:,1], marker='o',facecolors='none', 
              edgecolors='g', label='Class 1')
  plt.scatter(data[2][:,0], data[2][:,1], marker='s',facecolors='none', 
              edgecolors='b', label='Class 2')
  plt.legend()
  plt.xlabel(xlabel=r'input feature x1',size=15)
  plt.ylabel(ylabel=r'input feature x2',size=15)
  # plt.xticks(np.arange(len(xticks)),xticks)
  # plt.yticks(np.arange(len(yticks)),yticks)
  plt.show()

plot_decision_surface(params=params,dataset_no=dataset_no)

from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D

xx, yy = np.meshgrid(x,y)
pos = np.empty(xx.shape + (2,))
pos[:, :, 0] = xx; pos[:, :, 1] = yy
rv1 = multivariate_normal(params['mu'][0], params['var'][0])
rv2 = multivariate_normal(params['mu'][1], params['var'][1])
rv3 = multivariate_normal(params['mu'][2], params['var'][2])

plt.grid()
plt.title(r'Contour plots of best model for dataset {}'.format(dataset_no),size=15)
cs1 = plt.contour(xx, yy, rv1.pdf(pos),colors='r')
cs2 = plt.contour(xx, yy, rv2.pdf(pos),colors='g')
cs3 = plt.contour(xx, yy, rv3.pdf(pos),colors='b')

ev = np.zeros((c,d,d))
_,ev[0] = np.linalg.eig(params['var'][0])
_,ev[1] = np.linalg.eig(params['var'][1])
_,ev[2] = np.linalg.eig(params['var'][2])

plt.quiver(*params['mu'][0],ev[0][:,0],ev[0][:,1],color='r',width=0.002,scale=5)
plt.quiver(*params['mu'][1],ev[1][:,0],ev[1][:,1],color='g',width=0.002,scale=5) 
plt.quiver(*params['mu'][2],ev[2][:,0],ev[2][:,1],color='b',width=0.002,scale=5) 
plt.legend(['Class '+str(i) for i in range(3)])
plt.clabel(cs1, inline=True, fontsize=8)
plt.clabel(cs2, inline=True, fontsize=8)
plt.clabel(cs3, inline=True, fontsize=8)
plt.xlabel(r'input feature x1',size=15)
plt.ylabel(r'input feature x2',size=15)
# plt.xticks(np.arange(len(xticks)),xticks)
# plt.yticks(np.arange(len(yticks)),yticks)
plt.show()

#Make a 3D plot
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(xx, yy, rv1.pdf(pos),color='r',linewidth=0)
ax.plot_surface(xx, yy, rv2.pdf(pos),color='g',linewidth=0)
ax.plot_surface(xx, yy, rv3.pdf(pos),color='b',linewidth=0)
ax.set_xlabel(r'input feature x1$\rightarrow$')
ax.set_ylabel(r'input feature x2$\rightarrow$')
ax.set_zlabel(r'y$\rightarrow$')
# plt.xticks(np.arange(len(xticks)),xticks)
# plt.yticks(np.arange(len(yticks)),yticks)
plt.show()

