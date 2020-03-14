import numpy as np
import matplotlib.pyplot as plt
import random as rd
import sys



#  Generating highly correlated feature matrix ofsize 100 x 100 ( Columns are highly correlated )


dummy1 = np.zeros(100*100).reshape(100,100)
dummy2 = np.ones(100)
k = rd.uniform(0,0.1)
for i in range(len(dummy1)):
 for j in range(len(dummy1)):
  if i==j:
   dummy1[i][j] = k
  else:												
   dummy1[i][j] = rd.uniform(0.9,1)*k			# Making covariance very high
mean1 = np.multiply(dummy2,0.5)
cov1 = np.matmul(dummy1,dummy1.T)
Matrix1 = np.random.multivariate_normal(mean1,cov1,100)

Frobenius_norm_Matrix1 = np.linalg.norm(Matrix1)   # Finding Frobenius norm of Matrix
print("For HIGHLY CORRELATED MATRIX :-")
print("The Frobenius norm of the highly correlated matrix is " + str(Frobenius_norm_Matrix1))

a1 = np.matmul(Matrix1.T,Matrix1)
ev1,eve1=np.linalg.eig(a1)
evpa1 = [(ev1[i],eve1[:,i]) for i in range(len(ev1))]
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


e1 = np.matmul(q1.T[:,:10],np.matmul(n1[:10,:10],p1[:10,:]))
Fraction_frobenius_norm1 = np.linalg.norm(e1)/Frobenius_norm_Matrix1
print('Fraction of the Frobenius norm of A captured by the top 10 singular vectors is ' + str(Fraction_frobenius_norm1))


#  Fraction of the Frobenius norm of A captured by the random 10 singular vectors


p1_new = np.zeros(100*10).reshape(10,100)
q1_new = np.zeros(100*10).reshape(10,100)
n1_new = np.zeros(10*10).reshape(10,10)
for i in range(10): 
 j = rd.randrange(1,100)
 p1_new[i] = p1[j]
 q1_new[i] = q1[j]
 n1_new[i][i] = n1[j][j]
e1 = np.matmul(q1_new.T,np.matmul(n1_new,p1_new))
Fraction_frobenius_norm1 = np.linalg.norm(e1)/Frobenius_norm_Matrix1
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
print("Number of singular vectors required for capturing 50% of data is "  + str(r1[0]))

e1 = 0
for i in range(100):  # 75% of the data
 e1 += evpa1[i][0]
 if (e1**0.5)/Frobenius_norm_Matrix1 >= 0.75 :
  r1[1] = i+1
  break
print("Number of singular vectors required for capturing 75% of data is "  + str(r1[1]))

e1 = 0
for i in range(100):  # 95% of the data
 e1 += evpa1[i][0]
 if (e1**0.5)/Frobenius_norm_Matrix1 >= 0.95 :
  r1[2] = i+1
  break
print("Number of singular vectors required for capturing 95% of data is "  + str(r1[2]))

f1 = np.zeros(100)
g1 = np.array(range(1,101))
for i in range(1,len(Matrix1)+1):
 e1 = np.matmul(q1.T[:,:i],np.matmul(n1[:i,:i],p1[:i,:]))
 f1[i-1] = (np.linalg.norm(e1)/Frobenius_norm_Matrix1)*100

plt.plot(r1,s1,'o',color = 'red',label = 'Number of vectors carrying Y% of data')   # Required plot
plt.plot(g1,f1,'-',color = 'blue',label = 'Percentage of data carrired by X vectors')
plt.title(r'Number of singular vectors needed to capture Y% of data ( Highly correlated matrix ) ')
plt.legend(loc = 'best')
plt.xlabel(r'Number of singular vectors --> ')
plt.ylabel(r'Percentage of data captured --> ')
plt.grid(True)
plt.show()
print('')


#  Generating 100 x 100 matrix of statistically independent numbers between 0 and 1


Matrix2 = np.zeros(100*100).reshape(100,100)
for i in range(len(Matrix2)):
 for j in range(len(Matrix2[0])):
  rd.seed(rd.randint(1,1e10))
  Matrix2[i][j] = rd.uniform(0,1)

Frobenius_norm_Matrix2 = np.linalg.norm(Matrix2)   # Finding Frobenius norm of Matrix
print("For STATISTICALLY INDEPENDENT MATRIX :-")
print("The Frobenius norm of the statistically independent matrix is " + str(Frobenius_norm_Matrix2))

a2 = np.matmul(Matrix2.T,Matrix2)
ev2,eve2=np.linalg.eig(a2)
evpa2 = [(ev2.real[i],eve2[:,i]) for i in range(len(ev2))]
evpa2.sort()                                      # Finding eigen vectors and eigen values of Matrix and arranging them in descending order of eigen values
evpa2.reverse()		   
evpa2=np.array(evpa2)

p2 = np.zeros(len(Matrix2)*len(Matrix2)).reshape(len(Matrix2),len(Matrix2))    # p2 is the matrix made by eigen vectors of AT*A
q2 = np.zeros(len(Matrix2)*len(Matrix2)).reshape(len(Matrix2),len(Matrix2))    # q2 is the matrix made by eigen vectors of A*AT
n2 = np.zeros(len(Matrix2)*len(Matrix2)).reshape(len(Matrix2),len(Matrix2))    # n2 is the matrix made by sigma(square root of eigen values) of A
for i in range(len(Matrix2)):
 n2[i][i] = evpa2[i][0]**0.5
 p2[i] = evpa2[i][1]
 h2 = np.array([evpa2[i][1]]).T
 q2[i] = np.true_divide(np.matmul(Matrix2,h2),n2[i][i]).T[0]


#  Fraction of the Frobenius norm of A captured by the top 10 singular vectors


e2 = np.matmul(q2.T[:,:10],np.matmul(n2[:10,:10],p2[:10,:]))
Fraction_frobenius_norm2 = np.linalg.norm(e2)/Frobenius_norm_Matrix2
print('Fraction of the Frobenius norm of A captured by the top 10 singular vectors is ' + str(Fraction_frobenius_norm2))


#  Fraction of the Frobenius norm of A captured by the random 10 singular vectors

p2_new = np.zeros(100*10).reshape(10,100)
q2_new = np.zeros(100*10).reshape(10,100)
n2_new = np.zeros(10*10).reshape(10,10)
for i in range(10): 
 j = rd.randrange(1,100)
 p2_new[i] = p2[j]
 q2_new[i] = q2[j]
 n2_new[i][i] = n2[j][j]
e2 = np.matmul(q2_new.T,np.matmul(n2_new,p2_new))
e2_new = np.matmul(e2,e2.T)
Fraction_frobenius_norm2 = np.linalg.norm(e2)/Frobenius_norm_Matrix2
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
print("Number of singular vectors required for capturing 50% of data is "  + str(r2[0]))

e2 = 0
for i in range(100):     # 75% of the data
 e2 += evpa2[i][0]
 if (e2**0.5)/Frobenius_norm_Matrix2 >= 0.75 :
  r2[1] = i+1
  break
print("Number of singular vectors required for capturing 75% of data is "  + str(r2[1]))

e2 = 0
for i in range(100):      # 95% of the data
 e2 += evpa2[i][0]
 if (e2**0.5)/Frobenius_norm_Matrix2 >= 0.95 :
  r2[2] = i+1
  break
print("Number of singular vectors required for capturing 95% of data is "  + str(r2[2]))

f1 = np.zeros(100)
g1 = np.array(range(1,101))
for i in range(1,len(Matrix1)+1):
 e1 = np.matmul(q2.T[:,:i],np.matmul(n2[:i,:i],p2[:i,:]))
 f1[i-1] = (np.linalg.norm(e1)/Frobenius_norm_Matrix2)*100

plt.plot(r2,s2,'o',color = 'red',label = 'Number of vectors carrying Y% of data')   # Required plot
plt.plot(g1,f1,'-',color = 'blue',label = 'Percentage of data carrired by X vectors')
plt.title(r'Number of singular vectors needed to capture Y% of data ( Statistically independent matrix )')
plt.legend(loc = 'best')
plt.xlabel(r'Number of singular vectors --> ')
plt.ylabel(r'Percentage of data captured --> ')
plt.grid(True)
plt.show()
print('')

