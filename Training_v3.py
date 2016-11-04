import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import math
from pyalgotrade import strategy
from pyalgotrade import plotter
from pyalgotrade.tools import yahoofinance
from pyalgotrade.technical import vwap
from pyalgotrade.stratanalyzer import sharpe

from sklearn.preprocessing import normalize

from scipy.optimize import fmin_bfgs
from scipy.optimize import fmin_slsqp
from scipy.optimize import  fmin_ncg
import scipy.optimize as op

from numpy import asarray
from numpy import diag
from numpy import zeros_like
from numpy import genfromtxt

from UtilityFunctions import  GetReturns
from CoreFunctions_v3 import ObjectiveFunction
from CoreFunctions_v3 import GradientFunctionM
from CoreFunctions_v3 import  train
from CoreFunctions_v3 import ComputeF
from CoreFunctions_v3 import RewardFunction
from CoreFunctions_v3 import SharpeRatio
from UtilityFunctions import FeatureNormalize

#### MAIN ####


# Download the testing data from yahoo Finance.
#instruments = ["aapl"] # Set the instruments to download data.
#feed = yahoofinance.build_feed(instruments, 2011, 2012, "./Data") # Download the specified data.

# Retrieve data as an NP array.
trainingData = genfromtxt('./Data/aapl-2011-yahoofinance.csv', delimiter=',') # Read the data from CSV file.
trainingData = trainingData[1:, 6]  # Select the adjusted close

trainingData2012 = genfromtxt('./Data/aapl-2012-yahoofinance.csv', delimiter=',') # Read the data from CSV file.
trainingData2012 = trainingData2012[1:, 6]  # Select the adjusted close

#print(trainingData.size)
trainingData = np.concatenate([trainingData,trainingData2012],axis=0)
#print (trainingData.size)

# Load test training data
#trainingData = genfromtxt('./Data/test2.csv', delimiter=',') # Read the data from CSV file.
#trainingData = trainingData[:]  # Select the adjusted close

## Generate simulated signal
'''
media = 50
Line = np.ones(600) * media
deviation = np.random.random(600)*10
sig = np.add(Line,deviation)

t = np.linspace(0, 500, 500, endpoint=False)
sig = np.sin(20 * np.pi * t)
sig = signal.square(2 * np.pi * 30 * t, duty=(sig + 10)/2)
plt.plot(t, sig)
plt.show()
'''
# Set the parameters
M = 50
T = 50
startPos = M
finishPos = startPos + T
transactionCosts = 0.001
mu = 1
F = np.zeros(T + 1)
dF = np.zeros(T + 1)

# Generate the returns.
#X = trainingData
X = GetReturns(trainingData)
Xn = FeatureNormalize(X)

# Generate theta
theta = np.ones(M+2)

#g=GradientFunctionM(theta,X,Xn,T,M,mu,transactionCosts,startPos)
#print g
#print ObjectiveFunction(theta,X,Xn,T,M,mu,transactionCosts,startPos)
# Compute the output F.
#print("Computing neuron outputs F... ")
#F = ComputeF(theta, Xn, T, M, startPos)

# Compute the rewards
#print("Computing reward function... ")
#Rewards = RewardFunction(mu, F, transactionCosts, T, M, X)

# Compute the Sharpe ratio.
#print("Computing Sharpe ratio... ")
#sharpe = SharpeRatio(Rewards)

# Compute dFt.
#gradient = GradientFunctionM(theta, X, T, M, mu, transactionCosts, startPos)

#print(gradient, gradient.size)
#print(Rewards, Rewards.size)
#print(F, F.size)
#print(theta, theta.size)


print ("Computing theta...")
# Compute optimal theta and cost
#xopt = fmin_bfgs(ObjectiveFunction, x0=theta, args=(X[0:windowSize], mu, transactionCosts, M, T))
#theta = op.minimize(fun=ObjectiveFunction,x0=theta, jac=GradientFunctionM, args=(X, Xn, T, M, mu, transactionCosts, startPos), method='TNC')
#theta = fmin_ncg(ObjectiveFunction, theta, GradientFunctionM, args=(X, Xn, T, M, mu, transactionCosts, startPos), avextol=1e-16, maxiter=50)
theta = train(theta, X, Xn, T, M, mu, transactionCosts, startPos, 0.9, 500)
print ("Computing theta finished.")

# Get the optimal values for theta.
#theta = theta.x

#print theta
# Compute the output F.
print("Computing neuron outputs F... ")
F = ComputeF(theta, Xn, T, M, startPos)
# Compute the rewards
print("Computing reward function... ")
rewards = RewardFunction(mu, F, transactionCosts, T, M, X)
# Compute the cumulative reward.
rewards = rewards + 1 # Add one to the rewards vector such that the reward does not vanish.
for i in range(1, rewards.size):
    rewards[i] = rewards[i-1]*rewards[i]

returns = X[startPos:finishPos] + 1
for i in range(1, returns.size):
    returns[i] = returns[i-1]*returns[i]


# Plotting #
plt.subplot(3, 1, 1)
plt.title("Returns")
plt.plot(returns)
#plt.xlim([0, 210])
plt.subplot(3, 1, 2)
plt.title("Neuron Output")
plt.plot([0, T], [0, 0], color='k', linestyle='-', linewidth=2)
plt.plot(F[1:], color='r', linestyle='--')
plt.ylim([-1,1])


#plt.xlim([0, 210])
plt.subplot(3, 1, 3)
plt.title("Agent Rewards")
plt.plot(rewards)
#plt.xlim([0, 210])
plt.show()

fig, ax1 = plt.subplots()
t = np.arange(0.0, 150.0, 1)
ax1.plot(F[1:], 'y')
ax1.set_xlabel('Days')
# Make the y-axis label and tick labels match the line color.
ax1.set_ylabel('Neuron Output', color='b')
for tl in ax1.get_yticklabels():
    tl.set_color('b')

B = F[1:] > 0
for i,b in enumerate(B):
    if b:
        plt.plot([i, i], [-1, 1], color='b', linestyle='-', linewidth=2)
    else:
        plt.plot([i, i], [-1, 1], color='g', linestyle='-', linewidth=2)

ax2 = ax1.twinx()
ax2.plot(returns, 'r-', linewidth=3)
ax2.set_ylabel('Returns', color='r')
for tl in ax2.get_yticklabels():
    tl.set_color('r')


plt.show()

#### END MAIN ####
