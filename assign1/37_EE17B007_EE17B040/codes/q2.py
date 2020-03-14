"""# **Question 2**"""

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import pandas as pd

np.set_printoptions(suppress=True)

df = pd.read_csv('Dataset_3_Team_37.csv')
data = df['# x_1'].values
N = data.shape[0]

mu_0 = -1
r = 0.1
mu = (N/(N+r))*np.mean(data) + (r/(N+r))*mu_0
var = np.mean((data-mu)**2)
sigma = np.sqrt(var)

def plot_gaussian(mu=0,sigma=1,xlabel=None,ylabel=None,rate=-1,n=-1,figsize=(9,6)):
  x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
  plt.figure(figsize=figsize)
  plt.title(r'n = {}, $\sigma^2/\sigma_0^2 = {}, \mu = {:.4f}, \sigma = {:.4f}$'.format(n,rate,mu,sigma),size=15)
  plt.grid()
  plt.plot(x, stats.norm.pdf(x, mu, sigma))
  plt.xlabel(xlabel=r'x$\rightarrow$',size=15)
  plt.ylabel(ylabel=r'$y\rightarrow$',size=15)
  plt.savefig('image_'+str(n)+'_r_'+str(rate)+'.png')
  plt.show()

list_n = [10,100,1000]
list_r = [0.1,1,10,100]
for n in list_n:
  x = np.random.choice(data,size=n,replace=False)
  for r in list_r:
    mu_n = (n/(n+r))*np.mean(x) + (r/(n+r))*mu_0
    sigma_n = sigma/np.sqrt(n+r)
    plot_gaussian(mu=mu_n,sigma=sigma_n,xlabel='x',ylabel='y',rate=r,n=n)
