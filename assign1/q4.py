import numpy as np
import matplotlib.pyplot as plt
import random as rd
import sys



#  Generating highly correlated feature matrix ofsize 100 x 100 ( Columns are highly correlated )


Matrix1 = np.zeros(100*100).reshape(100,100)
for i in range(len(Matrix1)):
 for j in range(len(Matrix1[0])):
  Matrix1[i][j] = rd.uniform(0,1)

Frobenius_norm_Matrix1 = np.linalg.norm(Matrix1) # Finding Frobenius norm of Matrix
a1 = np.matmul(Matrix1.T,Matrix1)
ev1,eve1=np.linalg.eig(a1)
evpa1 = [(ev1.real[i],eve1[:,i]) for i in range(len(ev1))]
evpa1.sort()                          # Finding eigen vectors and eigen values of Matrix and arranging them in descending order of eigen values
evpa1.reverse()		   
evpa1=np.array(evpa1)

p1 = np.zeros(len(Matrix1)*len(Matrix1)).reshape(len(Matrix1),len(Matrix1))  # p1 is the matrix made by eigen vectors of AT*A  
q1 = np.zeros(len(Matrix1)*len(Matrix1)).reshape(len(Matrix1),len(Matrix1))  # q1 is the matrix made by eigen vectors of A*AT
n1 = np.zeros(len(Matrix1)*len(Matrix1)).reshape(len(Matrix1),len(Matrix1))  # n1 is the matrix made by sigma(square root of eigen values) of A
for i in range(len(Matrix1)):
 n1[i][i] = evpa1[i][0]**0.5
 p1[i] = evpa1[i][1]
 h1 = np.array([evpa1[i][1]]).T
 q1[i] = np.true_divide(np.matmul(Matrix1,h1),n1[i][i]).T[0]


#  Fraction of the Frobenius norm of A captured by the top 10 singular vectors


e1 = 0
for i in range(10):
 e1 += evpa1[i][0]
Fraction_frobenius_norm1 = (e1**0.5)/Frobenius_norm_Matrix1
print('Fraction of the Frobenius norm of A captured by the top 10 singular vectors is ' + str(Fraction_frobenius_norm1))


#  Fraction of the Frobenius norm of A captured by the random 10 singular vectors


e1 = 0
for i in range(10): 
 j = rd.randrange(0,100)
 e1 += evpa1[j][0]
Fraction_frobenius_norm1 = (e1**0.5)/Frobenius_norm_Matrix1
print('Fraction of the Frobenius norm of A captured by the random 10 singular vectors is ' + str(Fraction_frobenius_norm1))


#  Plot between number of singular vectors needed to capture 50% 75% 95% of data


e1 = 0
r1 = [0,0,0]
r1 = np.array(r1)
s1 = [50,75,95]
s1 = np.array(s1)
for i in range(100):   # 50% of the data
 e1 += evpa1[i][0]
 if (e1**0.5)/Frobenius_norm_Matrix1 >= 0.5 :
  r1[0] = i+1
  break

e1 = 0
for i in range(100):  # 75% of the data
 e1 += evpa1[i][0]
 if (e1**0.5)/Frobenius_norm_Matrix1 >= 0.75 :
  r1[1] = i+1
  break

e1 = 0
for i in range(100):  # 95% of the data
 e1 += evpa1[i][0]
 if (e1**0.5)/Frobenius_norm_Matrix1 >= 0.95 :
  r1[2] = i+1
  break

plt.plot(s1,r1,'o',color = 'red')   # Required plot
plt.title(r'Number of singular vectors needed to capture X% of data ( Highly correlated matrix ) ')
plt.xlabel(r'Percentage of data captured --> ')
plt.ylabel(r'Number of singular vectors --> ')
plt.grid(True)
plt.show()



#  Generating 100 x 100 matrix of statistically independent numbers between 0 and 1


Matrix2 = np.zeros(100*100).reshape(100,100)
for i in range(len(Matrix2)):
 for j in range(len(Matrix2[0])):
  Matrix2[i][j] = rd.uniform(0,1)

Frobenius_norm_Matrix2 = np.linalg.norm(Matrix2)   # Finding Frobenius norm of Matrix
a2 = np.matmul(Matrix2.T,Matrix2)
ev2,eve2=np.linalg.eig(a2)
evpa2 = [(ev2.real[i],eve2[:,i]) for i in range(len(ev2))]
evpa2.sort()                                      # Finding eigen vectors and eigen values of Matrix and arranging them in descending order of eigen values
evpa2.reverse()		   
evpa2=np.array(evpa2)

p2 = np.zeros(len(Matrix2)*len(Matrix2)).reshape(len(Matrix2),len(Matrix2))    # p1 is the matrix made by eigen vectors of AT*A
q2 = np.zeros(len(Matrix2)*len(Matrix2)).reshape(len(Matrix2),len(Matrix2))    # q1 is the matrix made by eigen vectors of A*AT
n2 = np.zeros(len(Matrix2)*len(Matrix2)).reshape(len(Matrix2),len(Matrix2))    # n1 is the matrix made by sigma(square root of eigen values) of A
for i in range(len(Matrix2)):
 n2[i][i] = evpa2[i][0]**0.5
 p2[i] = evpa2[i][1]
 h2 = np.array([evpa2[i][1]]).T
 q2[i] = np.true_divide(np.matmul(Matrix2,h2),n2[i][i]).T[0]


#  Fraction of the Frobenius norm of A captured by the top 10 singular vectors


e2 = 0
for i in range(10):
 e2 += evpa2[i][0]
Fraction_frobenius_norm2 = (e2**0.5)/Frobenius_norm_Matrix2
print('Fraction of the Frobenius norm of A captured by the top 10 singular vectors is ' + str(Fraction_frobenius_norm2))


#  Fraction of the Frobenius norm of A captured by the random 10 singular vectors


e2 = 0
for i in range(10): 
 j = rd.randrange(0,100)
 e2 += evpa2[j][0]
Fraction_frobenius_norm2 = (e2**0.5)/Frobenius_norm_Matrix2
print('Fraction of the Frobenius norm of A captured by the random 10 singular vectors is ' + str(Fraction_frobenius_norm2))


#  Plot between number of singular vectors needed to capture 50% 75% 95% of data


e2 = 0
r2 = [0,0,0]
r2 = np.array(r2)
s2 = [50,75,95]
s2 = np.array(s2)
for i in range(100):     # 50% of the data
 e2 += evpa2[i][0]
 if (e2**0.5)/Frobenius_norm_Matrix2 >= 0.5 :
  r2[0] = i+1
  break

e2 = 0
for i in range(100):     # 75% of the data
 e2 += evpa2[i][0]
 if (e2**0.5)/Frobenius_norm_Matrix2 >= 0.75 :
  r2[1] = i+1
  break

e2 = 0
for i in range(100):      # 95% of the data
 e2 += evpa2[i][0]
 if (e2**0.5)/Frobenius_norm_Matrix2 >= 0.95 :
  r2[2] = i+1
  break

plt.plot(s2,r2,'o',color = 'red')   # Required plot
plt.title(r'Number of singular vectors needed to capture X% of data ( Statistically independent matrix )')
plt.xlabel(r'Percentage of data captured --> ')
plt.ylabel(r'Number of singular vectors --> ')
plt.grid(True)
plt.show()

