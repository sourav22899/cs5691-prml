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


x = np.concatenate((x1,x2,x3))
mean = np.true_divide(np.sum(x,axis = 0),len(x))  # Estimating mean
x_new = x.copy()
for i in range(len(x)):
 x_new[i] = np.subtract(x[i],mean)
cov = np.true_divide(np.matmul(x_new.T,x_new),len(x))  # Estimating covariance matrix


#      Shrinking the covariance matrix for given training data and
#     Plotting the Training error Vs alpha for three distributions


alpha = np.true_divide(np.array(range(0,101)),100)   # Creating 100 points between 0 and 1 for alpha
training_error = np.zeros(303).reshape(3,101)
for i in range(len(alpha)):                          # Shrinking the covariances for each alpha using the given formula
 cov_updated1 = np.true_divide((np.multiply(cov1,(1-alpha[i])*len(x1))+np.multiply(cov,alpha[i]*len(x))),(1-alpha[i])*len(x1)+alpha[i]*len(x))
 cov_updated2 = np.true_divide((np.multiply(cov2,(1-alpha[i])*len(x2))+np.multiply(cov,alpha[i]*len(x))),(1-alpha[i])*len(x2)+alpha[i]*len(x))
 cov_updated3 = np.true_divide((np.multiply(cov3,(1-alpha[i])*len(x3))+np.multiply(cov,alpha[i]*len(x))),(1-alpha[i])*len(x3)+alpha[i]*len(x))
 error1 = np.subtract(cov1,cov_updated1)
 error2 = np.subtract(cov2,cov_updated2)
 error3 = np.subtract(cov3,cov_updated3)
 a1 = np.matmul(error1,error1.T)               # Finding norm of the error matrix and given covariance matrix for given distibutions
 a2 = np.matmul(error2,error2.T)
 a3 = np.matmul(error3,error3.T)
 b1 = np.matmul(cov1,cov1.T)
 b2 = np.matmul(cov2,cov2.T)
 b3 = np.matmul(cov3,cov3.T)                   # Training error can be defined as the norm of error matrix divided by norm of given covariance matrix
 training_error[0][i] = np.trace(a1)**0.5/np.trace(b1)**0.5
 training_error[1][i] = np.trace(a2)**0.5/np.trace(b2)**0.5
 training_error[2][i] = np.trace(a3)**0.5/np.trace(b3)**0.5 
 
plt.plot(alpha,training_error[0],'red',label='$\Sigma_{0}$ error')
plt.plot(alpha,training_error[1],'blue',label='$\Sigma_{1}$ error')   # Plotting the Training errors of three distributions Vs alpha
plt.plot(alpha,training_error[2],'green',label='$\Sigma_{2}$ error')
plt.legend(loc='upper left')
plt.title(r'Training error [= norm($\Sigma_{i}-\Sigma_{i}$($\alpha$))/norm($\Sigma_{i}$)] Vs Alpha')
plt.xlabel(r'Alpha -->')
plt.ylabel(r'Training Error --> ')
plt.grid(True)
plt.show()


#    Plotting Test error Vs alpha

                                         
x1_new = np.random.multivariate_normal(mean1,cov1,50) # Generating test data using the same method used to generate training data
x2_new = np.random.multivariate_normal(mean2,cov2,50)
x3_new = np.random.multivariate_normal(mean3,cov3,50)
                                          
x_new = np.concatenate((x1_new,x2_new,x3_new))
mean_new = np.true_divide(np.sum(x_new,axis = 0),len(x_new))
x_new_new = x_new.copy()                           # Estimating mean and covariance matrix for the given test data
for i in range(len(x_new)):                        #            using Maximum Likelihood Estimation
 x_new_new[i] = np.subtract(x_new[i],mean_new)
cov_new = np.true_divide(np.matmul(x_new_new.T,x_new_new),len(x_new))

test_error = np.zeros(303).reshape(3,101)
for i in range(len(alpha)):                        # Shrinking the covariances for each alpha using the given formula
 cov_updated1_new = np.true_divide((np.multiply(cov1,(1-alpha[i])*len(x1_new))+np.multiply(cov_new,alpha[i]*len(x_new))),(1-alpha[i])*len(x1_new)+alpha[i]*len(x_new))
 cov_updated2_new = np.true_divide((np.multiply(cov2,(1-alpha[i])*len(x2_new))+np.multiply(cov_new,alpha[i]*len(x_new))),(1-alpha[i])*len(x2_new)+alpha[i]*len(x_new))
 cov_updated3_new = np.true_divide((np.multiply(cov3,(1-alpha[i])*len(x3_new))+np.multiply(cov_new,alpha[i]*len(x_new))),(1-alpha[i])*len(x3_new)+alpha[i]*len(x_new))
 error4 = np.subtract(cov1,cov_updated1_new)
 error5 = np.subtract(cov2,cov_updated2_new)
 error6 = np.subtract(cov3,cov_updated3_new)
 a4 = np.matmul(error4,error4.T)                   # Finding norm of the error matrix and given covariance matrix for given distibution
 a5 = np.matmul(error5,error5.T)
 a6 = np.matmul(error6,error6.T)
 b4 = np.matmul(cov1,cov1.T)
 b5 = np.matmul(cov2,cov2.T)
 b6 = np.matmul(cov3,cov3.T) 
 test_error[0][i] = np.trace(a4)**0.5/np.trace(b4)**0.5    # Test error can be defined as the norm of error matrix divided by norm of given covariance matrix
 test_error[1][i] = np.trace(a5)**0.5/np.trace(b5)**0.5
 test_error[2][i] = np.trace(a6)**0.5/np.trace(b6)**0.5 
 
plt.plot(alpha,test_error[0],'red',label='$\Sigma_{0}$ error')
plt.plot(alpha,test_error[1],'blue',label='$\Sigma_{1}$ error')  # Plotting the Test errors of three distributions Vs alpha
plt.plot(alpha,test_error[2],'green',label='$\Sigma_{2}$ error')
plt.legend(loc='upper left')
plt.title(r'Test error [= norm($\Sigma_{i}-\Sigma_{i}$($\alpha$))/norm($\Sigma_{i}$)] Vs Alpha')
plt.xlabel(r'Alpha -->')
plt.ylabel(r'Test Error --> ')
plt.grid(True)
plt.show()

