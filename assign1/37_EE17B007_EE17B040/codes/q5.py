#Please name the file as 37.jpg

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random as rd

np.set_printoptions(suppress=True)

print('Please name the file as 37.jpg')
print(' ')

## Converting to gray scale


def rgb2gray(rgb):   # Used to convert rbg image to gray scale
 return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

img = mpimg.imread('37.jpg')  
gray = rgb2gray(img) 
plt.imshow(gray, cmap=plt.get_cmap('gray'))
plt.title(r'Grayscale image of the given picture',size=15)
plt.show()      # Plotting the grayscale picture

grayt=gray.T
gray_avg = [np.sum(i) for i in gray]   # Finding out the mean of all the data samples
gray_avg = np.true_divide(gray_avg,len(gray[0]))
gray_updated = gray.copy() 

for i in range(len(gray)):
 for j in range(len(gray[0])):         
  gray_updated[i][j] -= gray_avg[i]    # Getting the matrix A

gray_updatedt = gray_updated.T
y = np.matmul(gray_updated,gray_updatedt)
c = np.true_divide(y,len(gray[0]))        # Getting the matrix C=(1/n)ATA

ev,eve = np.linalg.eig(c)        # Finding out the eigen vectors of the matrix C
evpa = [(ev.real[i],eve[:,i]) for i in range(len(ev))]
evpa.sort()
evpa.reverse()		   # Arranging the eigen vectors in descending order of eigen values


## N = 10% (Top)


k1 = gray_updatedt.copy()
d1 = len(gray)//10            # N=10%
for j in range(len(gray[0])):    # Updating data points in the principal components
 for i in range(d1):
  k1[j] += np.multiply(evpa[i][1],np.dot(grayt[j],evpa[i][1]))
 for i in range(d1,len(gray)):
  k1[j] += np.multiply(evpa[i][1],np.dot(gray_avg,evpa[i][1]))
 k1[j] = np.subtract(k1[j],gray_updatedt[j])

q1 = k1.T       
plt.imshow(q1, cmap=plt.get_cmap('gray'))
plt.title(r'Reconstructed image using top 10% principal components ',size=15)
plt.show()       # Plotting the grayscale picture with N=10%

error1 = np.subtract(gray,q1)
plt.imshow(error1, cmap=plt.get_cmap('gray'))
plt.title(r'Error image using top 10% principal components',size=15)
plt.show()       # Plotting the error picture for N=10%

a1 = np.matmul(error1,error1.T)
b1 = np.matmul(q1,q1.T)
reconstruction_error1 = np.trace(a1)**0.5/np.trace(b1)**0.5
quality1 = (1-reconstruction_error1)*100      # Finding the quality of the reconsructed image
print("Quality of the image if we take top 10% principal components will be " + str(quality1) + ' %')


## N = 25% (Top)

k2 = gray_updatedt.copy()
d2 = len(gray)//4           # N=25%
for j in range(len(gray[0])):      # Updating data points in the principal components
 for i in range(d2):
  k2[j] += np.multiply(evpa[i][1],np.dot(grayt[j],evpa[i][1]))
 for i in range(d2,len(gray)):
  k2[j] += np.multiply(evpa[i][1],np.dot(gray_avg,evpa[i][1]))
 k2[j] = np.subtract(k2[j],gray_updatedt[j])

q2=k2.T
plt.imshow(q2, cmap=plt.get_cmap('gray'))
plt.title(r'Reconstructed image using top 25% principal components',size=15)
plt.show()        # Plotting the grayscale picture with N=25%

error2=np.subtract(gray,q2)
plt.imshow(error2, cmap=plt.get_cmap('gray'))
plt.title(r'Error image using top 25% principal components',size=15)
plt.show()        # Plotting the error picture for N=25%

a2 = np.matmul(error2,error2.T)
b2 = np.matmul(q2,q2.T)
reconstruction_error2 = np.trace(a2)**0.5/np.trace(b2)**0.5
quality2 = (1-reconstruction_error2)*100      # Finding the quality of the reconsructed image
print("Quality of the image if we take top 25% principal components will be " + str(quality2) + ' %')


## N = 50% (Top)


k3 = gray_updatedt.copy()
d3 = len(gray)//2        # N=50%
for j in range(len(gray[0])):      # Updating data points in the principal components
 for i in range(d3):
  k3[j] += np.multiply(evpa[i][1],np.dot(grayt[j],evpa[i][1]))
 for i in range(d3,len(gray)):
  k3[j] += np.multiply(evpa[i][1],np.dot(gray_avg,evpa[i][1]))
 k3[j] = np.subtract(k3[j],gray_updatedt[j])

q3=k3.T
plt.imshow(q3, cmap=plt.get_cmap('gray'))
plt.title(r'Reconstructed image using top 50% principal components',size=15)
plt.show()      # Plotting the grayscale picture with N=50%

error3=np.subtract(gray,q3)
plt.imshow(error3, cmap=plt.get_cmap('gray'))
plt.title(r'Error image using top 50% principal components',size=15)
plt.show()      # Plotting the error picture for N=50%

a3 = np.matmul(error3,error3.T)
b3 = np.matmul(q3,q3.T)
reconstruction_error3 = np.trace(a3)**0.5/np.trace(b3)**0.5
quality3 = (1-reconstruction_error3)*100       # Finding the quality of the reconsructed image
print("Quality of the image if we take top 50% principal components will be " + str(quality3) + ' %')


## N = 10% (Random)


k4 = gray_updatedt.copy()
d4 = len(gray)//10     # N=10%

for j in range(len(gray[0])):  # Updating data points in the principal components
 for i in range(0,len(gray)):
  k4[j] += np.multiply(evpa[i][1],np.dot(gray_avg,evpa[i][1]))
 for i in range(d4):
  p = rd.randrange(0,len(gray))  # Random generator
  k4[j] += np.multiply(evpa[p][1],np.dot(grayt[j],evpa[p][1]))
  k4[j] -= np.multiply(evpa[p][1],np.dot(gray_avg,evpa[p][1]))
 k4[j] = np.subtract(k4[j],gray_updatedt[j])

q4=k4.T
plt.imshow(q4, cmap=plt.get_cmap('gray'))
plt.title(r'Reconstructed image using random 10% principal components',size=15)
plt.show()       # Plotting the grayscale picture with N=10% (random)

error4=np.subtract(gray,q4)
plt.imshow(error4, cmap=plt.get_cmap('gray'))
plt.title(r'Error image using random 10% principal components',size=15)
plt.show()      # Plotting the error picture for N=10% (random)

a4 = np.matmul(error4,error4.T)
b4 = np.matmul(q4,q4.T)
reconstruction_error4 = np.trace(a4)**0.5/np.trace(b4)**0.5
quality4 = (1-reconstruction_error4)*100    # Finding the quality of the reconsructed image
print("Quality of the image if we take random 10% principal components will be " + str(quality4) + ' %')


## Reconstruction Error Vs N (where N goes form 10% to 100%)


reconstruction_error5 = np.zeros(10)
arr = np.multiply(np.array(range(1,11)),10)
for k in range(1,11):     # Finding 10 data points
 k5 = gray_updatedt.copy()
 d5 = (len(gray)*k)//10
 for j in range(len(gray[0])):   # Updating data points in the principal components
  for i in range(d5):
   k5[j] += np.multiply(evpa[i][1],np.dot(grayt[j],evpa[i][1]))
  for i in range(d5,len(gray)):
   k5[j] += np.multiply(evpa[i][1],np.dot(gray_avg,evpa[i][1]))
  k5[j] = np.subtract(k5[j],gray_updatedt[j])
  
 q5=k5.T
 error5=np.subtract(gray,q5)  
 a5 = np.matmul(error5,error5.T)
 b5 = np.matmul(q5,q5.T)     # Finding the reconstruction error
 reconstruction_error5[k-1] = np.trace(a5)**0.5/np.trace(b5)**0.5
 
plt.plot(arr,reconstruction_error5,'red',marker='o')
plt.title(r'Reconstruction Error Vs N (where N goes form 10% to 100%)',size=15)
plt.xlabel(r'Top N% principal components are taken $\rightarrow$')
plt.ylabel(r'Reconstruction Error $\rightarrow$')
plt.grid(True)
plt.show()
