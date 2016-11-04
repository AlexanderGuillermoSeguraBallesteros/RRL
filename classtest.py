import matplotlib.pyplot as plt
# Import libraries for time series acquisition.
import quandl
# Import the required classes for Recurrent Reinforcement Learning
from RRL.rrl_class import RRL as rrl
# Import math and scientific processing libraries.
import numpy as np
from scipy.optimize import fmin_ncg
# Import utility functions
from RRL.UtilityFunctions import GetReturns
from RRL.UtilityFunctions import FeatureNormalize


# Get training and testing data from quandl.
# Configure quandl account.
quandl.ApiConfig.api_key = "UYzP-62GzccS1s_3NToj"
trainingData = quandl.get("WIKI/AAPL", start_date="2008-01-01", end_date="2014-12-31", returns="numpy")
trainingData = trainingData["Adj. Close"]
testData = quandl.get("WIKI/AAPL", start_date="2014-01-01", end_date="2015-12-31", returns="numpy")
testData = testData["Adj. Close"]

# Set the parameters.
# Set the parameters
optimizationFunction = "return"
inputs_window = 50
training_window = 500
transactionCosts = 0.001
number_of_shares = 1
initial_theta = np.ones(inputs_window + 2)
X = GetReturns(trainingData)
Xn = FeatureNormalize(X)
startPos = inputs_window + training_window
finishPos = startPos + training_window

my_rrl = rrl(initial_theta=initial_theta, T=training_window, M=inputs_window,
             data=trainingData, optimization_function=optimizationFunction, rho=transactionCosts,
             mu=number_of_shares)
#my_rrl.update_theta(inputs_window, training_window, trainingData)
#print(my_rrl.theta)
#my_rrl.plot_decision()

'''
## PLOTTING ##
# Compute the output F.
print("Computing neuron outputs F... ")
F = my_rrl.__compute_F__(theta, Xn, training_window, inputs_window, startPos)
# Compute the rewards
print("Computing reward function... ")
rewards = my_rrl.reward_function(F, training_window, inputs_window, X)
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
plt.plot([0, training_window], [0, 0], color='k', linestyle='-', linewidth=2)
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
'''