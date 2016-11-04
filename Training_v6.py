# Import plot libraries.
import matplotlib.pyplot as plt

# Import Scipy functions
from scipy.optimize import fmin_bfgs
from scipy.optimize import fmin_slsqp
from scipy.optimize import  fmin_ncg
import scipy.optimize as op

# Import Numpy functions
import numpy as np
from numpy import genfromtxt

import quandl

# Import local functions.
from RRL.UtilityFunctions import  GetReturns
from RRL.CoreFunctions_v6 import ObjectiveFunction
from RRL.CoreFunctions_v6 import GradientFunctionM
from RRL.CoreFunctions_v6 import  train
from RRL.CoreFunctions_v6 import ComputeF
from RRL.CoreFunctions_v6 import RewardFunction
from RRL.CoreFunctions_v6 import SharpeRatio
from RRL.UtilityFunctions import FeatureNormalize
from RRL.rrl_class import RRL

#### MAIN ####


# Download the testing data from yahoo Finance.
#instruments = ["aapl"] # Set the instruments to download data.
#feed = yahoofinance.build_feed(instruments, 2011, 2012, "./Data") # Download the specified data.

# Retrieve data as an NP array.
#trainingData = genfromtxt('./Data/aapl-2011-yahoofinance.csv', delimiter=',') # Read the data from CSV file.
#trainingData = trainingData[1:, 6]  # Select the adjusted close

#testData = genfromtxt('./Data/aapl-2012-yahoofinance.csv', delimiter=',') # Read the data from CSV file.
#testData = testData[1:, 6]  # Select the adjusted close

#print(trainingData.size)
#trainingData = np.concatenate([trainingData,trainingData2012],axis=0)
#print (trainingData.size)

# Load test training data
#testData = genfromtxt('./Data/test2.csv', delimiter=',') # Read the data from CSV file.
#testData = testData[:]  # Select the adjusted close

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

# Get training and testing data from quandl.
# Configure quandl account.
quandl.ApiConfig.api_key = "UYzP-62GzccS1s_3NToj"
trainingData = quandl.get("WIKI/AAPL", start_date = "2008-01-01", end_date="2014-12-31", returns="numpy")
trainingData = trainingData["Adj. Close"]
testData = quandl.get("WIKI/AAPL", start_date = "2014-01-01", end_date="2015-12-31", returns="numpy")
testData = testData["Adj. Close"]

# Set the parameters
optimizationFunction = "return"
M = 50
T = 500
startPos = M
finishPos = startPos + T
transactionCosts = 0.001
mu = 1
F = np.zeros(T + 1)
dF = np.zeros(T + 1)

# Generate the returns.
# X = trainingData
X = GetReturns(trainingData)
Xn = FeatureNormalize(X)

returns = X[startPos:finishPos] + 1
for i in range(1, returns.size):
    returns[i] = returns[i-1]*returns[i]

# Generate theta
theta = np.ones(M+2)

'''
rrl = RRL(optimization_function=optimizationFunction)
print("Computing neuron outputs F... ")
F = ComputeF(theta, Xn, T, M, startPos)
print(rrl.__compute_F__(theta,Xn,T,M,startPos))
print(F)
print(sum(rrl.__compute_F__(theta,Xn,T,M,startPos)-F))
# Compute the rewards
print("Computing reward function... ")
rewards = RewardFunction(mu, F, transactionCosts, T, M, X)
print(sum(rrl.reward_function(rrl.__compute_F__(theta,Xn,T,M,startPos),T,M,X) - rewards))

print("objective function")
print(ObjectiveFunction(theta, X, Xn, T, M, mu, transactionCosts, startPos, optimizationFunction))
print(rrl.__ObjectiveFunction__(theta,X, Xn, T, M, startPos))

print("Gradient")
g1 = GradientFunctionM(theta, X, Xn, T, M, mu, transactionCosts, startPos, optimizationFunction)
g2 = rrl.__GradientFunctionM__(theta,X, Xn, T, M, startPos)
print(sum(g1-g2))
'''
print ("Computing theta...")
# Compute optimal theta and cost
theta = fmin_ncg(ObjectiveFunction, theta, GradientFunctionM, args=(X, Xn, T, M, mu, transactionCosts, startPos, optimizationFunction), avextol=1e-8, maxiter=50)
#theta = train(theta, X, Xn, T, M, mu, transactionCosts, startPos, 1, 500, optimizationFunction)
#theta2 = fmin_ncg(rrl.__ObjectiveFunction__, theta, rrl.__GradientFunctionM__,
#                 args=(X, Xn, T, M, startPos), avextol=1e-8, maxiter=50)
#)

## PLOTTING ##
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

plt.subplot(3, 1, 1)
plt.title("Returns")
plt.plot(returns)
plt.xlim([0, finishPos])
plt.subplot(3, 1, 2)
plt.title("Neuron Output")
plt.plot([0, T], [0, 0], color='k', linestyle='-', linewidth=2)
plt.plot(F[1:], color='r', linestyle='--')
plt.ylim([-1,1])
plt.xlim([0, finishPos])
plt.subplot(3, 1, 3)
plt.title("Agent Rewards")
plt.plot(rewards)
plt.xlim([0, finishPos])
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
## END PLOTTING ##


# Start the training #
# Set training window size.
N = 100
# Set inputs window size.
J = M
# Set number of training iterations.
I = 15
# Set the starting position.
startPos += 1

print("Training ...")
for i in range(1, I):
    print("On iteration: ", i)
    theta = fmin_ncg(ObjectiveFunction, theta, GradientFunctionM,
                     args=(X, Xn, N, J, mu, transactionCosts, startPos, optimizationFunction),
                     maxiter=50)
    #theta = train(theta, X, Xn, N, J, mu, transactionCosts, startPos, 1, 500, optimizationFunction)
    startPos += 1
print("Finished training.")

# Compute the output F.
print("Computing neuron outputs F... ")
F = ComputeF(theta, Xn, N, J, startPos)
# Compute the rewards
print("Computing reward function... ")
rewards = RewardFunction(mu, F, transactionCosts, N, J, X)
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


# Test
print (theta)
Xtest = testData
Xtest = GetReturns(testData)
Xtest_n = FeatureNormalize(Xtest)
M = 50
T = 50
startPos = M+T
# Compute the output F.
print("Computing neuron outputs F... ")
F = ComputeF(theta, Xtest_n, T, M, startPos)
# Compute the rewards
print("Computing reward function... ")
rewards = RewardFunction(mu, F, transactionCosts, T, M, Xtest)
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
