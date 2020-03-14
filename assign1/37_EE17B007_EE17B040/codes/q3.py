import numpy as np
import matplotlib.pyplot as plt

mean1 = [0,0,0]
cov1 = [[3,0,0],[0,5,0],[0,0,2]]
cov1 = np.array(cov1)			        # Generating 20 points from gaussian distribution 1
x1 = np.random.multivariate_normal(mean1,cov1,20)

mean2 = [1,5,-3]
cov2 = [[1,0,0],[0,4,1],[0,1,6]]
cov2 = np.array(cov2)             # Generating 20 points from gaussian distribution 2
x2 = np.random.multivariate_normal(mean2,cov2,20)

mean3 = [0,0,0]
cov3 = [[10,0,0],[0,10,0],[0,0,10]]
cov3 = np.array(cov3)             # Generating 20 points from gaussian distribution 3
x3 = np.random.multivariate_normal(mean3,cov3,20)


# Estimating mean and covariance matrix for the given training data
#            using Maximum Likelihood Estimation

print("For TRAINING DATA :-")
x = np.concatenate((x1,x2,x3))
mean = np.true_divide(np.sum(x,axis = 0),len(x))  # Estimating mean
print("Mean of the Training data estimated using Maximum likelihood estimation is " + str(mean))
x_new = x.copy()
for i in range(len(x)):
 x_new[i] = np.subtract(x[i],mean)
cov = np.true_divide(np.matmul(x_new.T,x_new),len(x))  # Estimating covariance matrix
print("Covariance matrix of the Training data estimated using Maximum likelihood estimation is")
print(cov)

# Multivariate gaussian likelihood

def gaussian_probability(mean,cov,x):
 p = np.subtract(x,mean)
 q = np.array([p])
 r = q.T
 y = (2*np.pi)**(len(x)/2.0)
 z = np.abs(np.linalg.det(cov))**0.5
 h = (-0.5)*(np.matmul(q,np.matmul(np.linalg.inv(cov),r))[0][0])
 return (1.0/(y*z))*np.exp(h)


#      Shrinking the covariance matrix for given training data and
#     Plotting the Training error Vs alpha for three distributions


alpha = np.true_divide(np.array(range(0,100)),100)   # Creating 100 points between 0 and 1 for alpha
training_error = np.zeros(100)
for i in range(len(alpha)):                          # Shrinking the covariances for each alpha using the given formula
 cov_updated1 = np.true_divide((np.multiply(cov1,(1-alpha[i])*len(x1))+np.multiply(cov,alpha[i]*len(x))),(1-alpha[i])*len(x1)+alpha[i]*len(x))
 cov_updated2 = np.true_divide((np.multiply(cov2,(1-alpha[i])*len(x2))+np.multiply(cov,alpha[i]*len(x))),(1-alpha[i])*len(x2)+alpha[i]*len(x))
 cov_updated3 = np.true_divide((np.multiply(cov3,(1-alpha[i])*len(x3))+np.multiply(cov,alpha[i]*len(x))),(1-alpha[i])*len(x3)+alpha[i]*len(x))
 for j in range(len(x1)):
  p = gaussian_probability(mean1,cov_updated1,x1[j])
  q = gaussian_probability(mean2,cov_updated2,x1[j])  
  r = gaussian_probability(mean3,cov_updated3,x1[j])
  if p>q and p>r:
   training_error[i] += 0
  else :
   training_error[i] += 1                              # Finding out the error for training data
 for j in range(len(x2)):
  p = gaussian_probability(mean1,cov_updated1,x2[j])
  q = gaussian_probability(mean2,cov_updated2,x2[j])
  r = gaussian_probability(mean3,cov_updated3,x2[j])
  if q>p and q>r:
   training_error[i] += 0
  else :
   training_error[i] += 1
 for j in range(len(x3)):
  p = gaussian_probability(mean1,cov_updated1,x3[j])
  q = gaussian_probability(mean2,cov_updated2,x3[j])
  r = gaussian_probability(mean3,cov_updated3,x3[j])
  if r>p and r>q:
   training_error[i] += 0
  else :
   training_error[i] += 1
 training_error[i] /= len(x)
 
plt.plot(alpha,training_error,'red',label = 'Number of Misclassified points divided by Total number of points')   # Plotting the Training error Vs alpha
plt.legend(loc = 'upper left')
plt.title(r'Training error Vs Alpha')
plt.xlabel(r'Alpha -->')
plt.ylabel(r'Training Error --> ')
plt.grid(True)
plt.show()
print('')


#    Plotting Test error Vs alpha

                                         
x1_new = np.random.multivariate_normal(mean1,cov1,50) # Generating test data using the same method used to generate training data
x2_new = np.random.multivariate_normal(mean2,cov2,50)
x3_new = np.random.multivariate_normal(mean3,cov3,50)
            
print("For TEST DATA :-")                              
x_new = np.concatenate((x1_new,x2_new,x3_new))
mean_new = np.true_divide(np.sum(x_new,axis = 0),len(x_new))
print("Mean of the Test data estimated using Maximum likelihood estimation is " + str(mean_new))
x_new_new = x_new.copy()                           # Estimating mean and covariance matrix for the given test data
for i in range(len(x_new)):                        #            using Maximum Likelihood Estimation
 x_new_new[i] = np.subtract(x_new[i],mean_new)
cov_new = np.true_divide(np.matmul(x_new_new.T,x_new_new),len(x_new))
print("Covariance matrix of the Test data estimated using Maximum likelihood estimation is")
print(cov_new)

test_error = np.zeros(100)
for i in range(len(alpha)):                        # Shrinking the covariances for each alpha using the given formula
 cov_updated1_new = np.true_divide((np.multiply(cov1,(1-alpha[i])*len(x1_new))+np.multiply(cov_new,alpha[i]*len(x_new))),(1-alpha[i])*len(x1_new)+alpha[i]*len(x_new))
 cov_updated2_new = np.true_divide((np.multiply(cov2,(1-alpha[i])*len(x2_new))+np.multiply(cov_new,alpha[i]*len(x_new))),(1-alpha[i])*len(x2_new)+alpha[i]*len(x_new))
 cov_updated3_new = np.true_divide((np.multiply(cov3,(1-alpha[i])*len(x3_new))+np.multiply(cov_new,alpha[i]*len(x_new))),(1-alpha[i])*len(x3_new)+alpha[i]*len(x_new))
 for j in range(len(x1_new)):
  p = gaussian_probability(mean1,cov_updated1_new,x1_new[j])
  q = gaussian_probability(mean2,cov_updated2_new,x1_new[j])
  r = gaussian_probability(mean3,cov_updated3_new,x1_new[j])
  if p>q and p>r:
   test_error[i] += 0
  else :
   test_error[i] += 1                               # Finding out the error for test data
 for j in range(len(x2_new)):
  p = gaussian_probability(mean1,cov_updated1_new,x2_new[j])
  q = gaussian_probability(mean2,cov_updated2_new,x2_new[j])
  r = gaussian_probability(mean3,cov_updated3_new,x2_new[j])
  if q>p and q>r:
   test_error[i] += 0
  else :
   test_error[i] += 1
 for j in range(len(x3_new)):
  p = gaussian_probability(mean1,cov_updated1_new,x3_new[j])
  q = gaussian_probability(mean2,cov_updated2_new,x3_new[j])
  r = gaussian_probability(mean3,cov_updated3_new,x3_new[j])
  if r>p and r>q:
   test_error[i] += 0
  else :
   test_error[i] += 1
 test_error[i] /= len(x_new)
 
plt.plot(alpha,test_error,'red',label = 'Number of Misclassified points divided by Total number of points')  # Plotting the Test error Vs alpha
plt.legend(loc = 'upper left')
plt.title(r'Test error Vs Alpha')
plt.xlabel(r'Alpha -->')
plt.ylabel(r'Test Error --> ')
plt.grid(True)
plt.show()
print('')

