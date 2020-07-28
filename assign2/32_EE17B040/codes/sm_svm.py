"""# Soft margin SVM"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split


data = pd.read_csv('Dataset_2_Team_32.csv')

y = data['Class_label']
X = data.drop(columns='Class_label')

X = np.asarray(X,dtype=np.float)
y = np.asarray(y,dtype=np.float)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print(X_train.shape)
print(X_test.shape)

C_list = [10**i for i in range(-4,5)] # C_list

for c in C_list:
    clf = SVC(kernel='linear',C=c)
    clf.fit(X_train, y_train)
    print('C:',c)
    print(clf.score(X_train,y_train))
    print(clf.score(X_test,y_test))
    print('-'*50)

# Plot Confusion Matrix

clf = SVC(kernel='linear',C=100000)
clf.fit(X_train, y_train)
plot_confusion_matrix(clf, X_test, y_test, values_format='.2g',cmap='gray')
plt.show()

# Plot decision boundary with margin hyperplanes

w = clf.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(-2.5, 2.5)
yy = a * xx - (clf.intercept_[0]) / w[1]

    # plot the parallels to the separating hyperplane that pass through the
    # support vectors (margin away from hyperplane in direction
    # perpendicular to hyperplane). This is sqrt(1+a^2) away vertically in
    # 2-d.
margin = 1 / np.sqrt(np.sum(clf.coef_ ** 2))
yy_down = yy - np.sqrt(1 + a ** 2) * margin
yy_up = yy + np.sqrt(1 + a ** 2) * margin

    # plot the line, the points, and the nearest vectors to the plane
plt.figure(0,figsize=(12,9))
plt.clf()
plt.plot(xx, yy, 'k-')
plt.plot(xx, yy_down, 'k--')
plt.plot(xx, yy_up, 'k--')

plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=80,
                facecolors='none', zorder=10, edgecolors='k')
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, zorder=10, cmap=plt.cm.Paired,
                edgecolors='k')

plt.axis('tight')
x_min = -1.5
x_max = 2.4
y_min = -3
y_max = 3

XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
Z = clf.predict(np.c_[XX.ravel(), YY.ravel()])

    # Put the result into a color plot
Z = Z.reshape(XX.shape)
plt.figure(0,figsize=(9,6))
plt.title('Decision boundary plot for Dataset 2 for C = 1',size=15)
plt.xlabel(r'$x_1\rightarrow$',size=15)
plt.ylabel(r'$x_2\rightarrow$',size=15)
plt.pcolormesh(XX, YY, Z, cmap=plt.cm.autumn)
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.show()

sv = clf.support_vectors_
wt = clf.coef_
b = clf.intercept_

# Find number of support vectors lying on margin hyperplanes

zx = np.matmul(X_train,wt.T)+b
la = zx >= 0.95
lb = zx <= 1.05
print('Postive hyperplane sv:', np.sum(la*lb))

zx = np.matmul(X_train,wt.T)+b
la = zx >= -1.05
lb = zx <= -0.95
print('Negative hyperplane sv:', np.sum(la*lb))

# Weighted loss SVM

tr_res, te_res = [], []
x = [1,2,4,8,16]
for i in range(5):
    clf = SVC(kernel='linear',class_weight={0:2**i,1:1})
    clf.fit(X_train, y_train)
    tr_res.append(clf.score(X_train,y_train))
    te_res.append(clf.score(X_test,y_test))
plt.figure(0,figsize=(9,6))
plt.plot(x,tr_res,'-+',label='train_error')
plt.plot(x,te_res,'-*',label='test_error')
plt.title('Train and test accuracies vs k', size=15)
plt.xlabel(r'$k\rightarrow$', size=15)
plt.ylabel(r'$accuracy\rightarrow$',size=15)
plt.grid()
plt.legend()
plt.show()
