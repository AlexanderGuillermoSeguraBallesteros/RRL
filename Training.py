import numpy as np
import matplotlib.pyplot as plt
import math
from pyalgotrade import strategy
from pyalgotrade import plotter
from pyalgotrade.tools import yahoofinance
from pyalgotrade.technical import vwap
from pyalgotrade.stratanalyzer import sharpe

from scipy.optimize import fmin_bfgs
from scipy.optimize import fmin_slsqp
from scipy.optimize import  fmin_ncg
import scipy.optimize as op

from numpy import asarray
from numpy import diag
from numpy import zeros_like
from numpy import genfromtxt

from UtilityFunctions import  GetReturns
from CoreFunctions import ObjectiveFunction
from CoreFunctions import ComputeF
from CoreFunctions import RewardFunction
from CoreFunctions import SharpeRatio
#### MAIN ####


# Download the testing data from yahoo Finance.
instruments = ["aapl"] # Set the instruments to download data.
feed = yahoofinance.build_feed(instruments, 2011, 2012, "./Data") # Download the specified data.

# Retrieve data as an NP array.
trainingData = genfromtxt('./Data/aapl-2011-yahoofinance.csv', delimiter=',') # Read the data from CSV file.
trainingData = trainingData[1:, 6]  # Select the adjusted close

trainingData2012 = genfromtxt('./Data/aapl-2012-yahoofinance.csv', delimiter=',') # Read the data from CSV file.
trainingData2012 = trainingData2012[1:, 6]  # Select the adjusted close

#print(trainingData.size)
trainingData = np.concatenate([trainingData,trainingData2012],axis=0)
#print (trainingData.size)

# Set the parameters
M = 10
T = 150
windowSize = T + M
transactionCosts = 0.0001
mu = 1

# Generate the returns.
X = GetReturns(trainingData)

# Generate theta
theta = np.ones(M+2)


# Compute the output F.
print("Computing neuron outputs F... ")
F = ComputeF(X[0:windowSize],theta,T,M)
# Compute the rewards
print("Computing reward function... ")
Rewards = RewardFunction(mu,F,transactionCosts,T,M,X[0:windowSize])
# Compute the Sharpe ratio.
print("Computing Sharpe ratio... ")
sharpe = SharpeRatio(Rewards)

#print(sharpe)
#print(Rewards, Rewards.size)
#print(F, F.size)
#print(theta, theta.size)


print ("Computing theta...")
# Compute optimal theta and cost
#xopt = fmin_bfgs(ObjectiveFunction, x0=theta, args=(X[0:windowSize], mu, transactionCosts, M, T))
theta = op.minimize(fun=ObjectiveFunction,x0=theta, args=(X[0:windowSize], mu, transactionCosts, M, T), method='TNC')
print ("Computing theta finished.")

# Get the optimal values for theta.
theta = theta.x

# Compute the output F.
print("Computing neuron outputs F... ")
F = ComputeF(X[0:windowSize], theta, T, M)
# Compute the rewards
print("Computing reward function... ")
rewards = RewardFunction(mu, F, transactionCosts, T, M, X[0:windowSize])
# Compute the cumulative reward.
rewards = rewards + 1 # Add one to the rewards vector such that the reward does not vanish.
for i in range(1,rewards.size):
    rewards[i] = rewards[i-1]*rewards[i]

# Plotting #

plt.subplot(3,1,1)
plt.plot(trainingData)
plt.xlim([0,210])
plt.subplot(3,1,2)
plt.plot(F[1:])
plt.xlim([0,210])
plt.subplot(3,1,3)
plt.plot(rewards)
plt.xlim([0,210])
plt.show()

## TRAINING ##
J = 10
N = 15
F = np.zeros(J*N)
rewards = np.zeros(J*N)
for j in range(0,J):
    print("Training ",j )
    # Compute the outputs for current theta.
    print("Computing neuron outputs F for training " )
    F_j = ComputeF(X[T + j*N:(j+1)*N+windowSize], theta, N, M)
    #F_j = ComputeF(X[T+i*N-1:(i+1)*N-1+T], theta, T, N)
    F[j*N:(j+1)*N] = F_j[1:]

    # Compute rewards
    rewards_j = RewardFunction(mu, F_j, transactionCosts, N, M, X[T + j*N:(j+1)*N+windowSize])
    rewards_j = rewards_j + 1
    rewards[j*N:(j+1)*N] = rewards_j

    for k in range(j*N,(j+1)*N):
        #print(k)
        if (k-1) > 0:
            rewards[k] = rewards[k-1]*rewards[k]

    print ("Computing theta for training",j)
    theta = op.minimize(fun=ObjectiveFunction, x0=theta, args=(X[T + j*N:(j+1)*N+windowSize], mu, transactionCosts, M, N), method='TNC')
    theta = theta.x

plt.subplot(3, 1, 1)
plt.plot(trainingData[T + M:J*N+windowSize])
plt.xlim([0, 210])
plt.subplot(3, 1, 2)
plt.plot(F[1:])
plt.xlim([0, 210])
plt.subplot(3, 1, 3)
plt.plot(rewards)
plt.xlim([0, 210])
plt.show()
## END TRAINING ##


#### END MAIN ####