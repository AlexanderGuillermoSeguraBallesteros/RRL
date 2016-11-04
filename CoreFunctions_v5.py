
'''
This file contains the core functions for the basic RRL algorithm. Two measures of performance are applied to the
algorithm: Returns and Sharpe Ratio. The reward function is implemented. considering transaction costs.
'''
import numpy as np
import math
from pyalgotrade import strategy
from pyalgotrade import plotter
from pyalgotrade.tools import yahoofinance
from pyalgotrade.technical import vwap
from pyalgotrade.stratanalyzer import sharpe
from scipy.optimize import fmin_bfgs
from numpy import asarray
from numpy import diag
from numpy import zeros_like


#### CORE FUNCTIONS ####

## REWARD FUNCTION ##
# Function to get the transaction cost at a time step t.
# F -> Vector containing the discrete output Ft at each time t from 1 to T + 1.
# X -> Vector containing the returns for the whole series.
def RewardFunction (mu, F, transactionCosts, T, M, X):
    #return mu*(F[0:T]*X[M:M+T-1] - transactionCosts*np.sign(F[1:T-1]-F[0:T-2]))
    F = np.ceil(np.absolute(F)) * np.sign(F)

    return mu*(F[0:T]*X[M:M+T] - transactionCosts*np.abs(F[1:T+1]-F[0:T]))
## END REWARD FUNCTION ##

## SHARPE RATIO ##
def SharpeRatio(Reward):
    Reward = np.asarray(Reward)
    A = Reward.mean()
    B = np.square(Reward).mean()
    return A/math.sqrt(B-math.pow(A, 2))
## END SHARPE RATIO ##


## COMPUTE F ##
# Returns all the values of F from time 0 to T+1.
def ComputeF(theta, X, T, M, startPos):
    F = np.zeros(T+1)
    for i in range(1, T+1):
        inputsWindow = X[(i-1) + startPos - M: startPos + i - 1]
        neuronInput = np.concatenate(([1], inputsWindow, [F[i-1]]), axis=0)
        F[i] = math.tanh(np.dot(theta, neuronInput))

    return F
## END COMPUTE F ##


## UTILITY FUNCTION ##
def ObjectiveFunction(theta, X, Xn, T, M, mu, transactionCost, startPos):
    # Compute the Sharpe ratio #
    F = ComputeF(theta, Xn, T, M, startPos) # Compute the values of vector of with current theta values.
    #print(F,F.size)
    Reward = RewardFunction(mu, F, transactionCost, T, M, X)
    sharpe = SharpeRatio(Reward)
    sharpe = -1*sharpe;
    A = Reward.mean()
    return -A

## END UTILITY FUNCTION ##

## GRADIENT ##
def GradientFunctionM(theta, X, Xn, T, M, mu, transactionCosts, startPos):

    F = ComputeF(theta, Xn, T, M, startPos)

    rewards = RewardFunction(mu, F, transactionCosts, T, M, X)

    # Initialize the vector dFt as a matrix of zeros. (The gradient has a component per input).
    gradient = np.zeros([M + 2, T+1])
    dFt = np.zeros([M+2, T + 1])
    #dRtdFt = np.zeros(T)
    #dRtdFtp = np.zeros(T)

    # Compute the derivative of F as a vector (dFt).
    for i in range(1,T+1):
        inputsWindow = Xn[(i - 1) + startPos - M: startPos + i - 1]
        neuronInput = np.concatenate(([1], inputsWindow, [F[i - 1]]), axis=0)
        dFt[:, i] = (1 - math.pow(math.tanh(np.dot(theta, neuronInput)), 2))*(neuronInput + theta[M+1]*dFt[:, i-1])

    A = rewards.mean() # Compute the mean.
    B = np.square(rewards).mean() # Compute the stardar deviation.

    A2 = math.pow(A,2)
    under = math.pow(B-A2,3/2)

    #dStdA = B/math.pow((B-math.pow(A, 2)), (3/2))
    #dStdB = -0.5*(A/math.pow((B-math.pow(A,2)),(3/2)))
    dStdA = B / under
    dStdB = -A /2*under
    dAdRt = 1
    dBdRt = 2*A
    dRtdFt = -mu * transactionCosts * np.sign(F[1:T+1] - F[0:T])
    dRtdFtp = mu * X[M:M+T] + mu * transactionCosts * np.sign(F[1:T+1] - F[0:T])
    #gradient = (dStdA  + dStdB) * (dRtdFt * dFt[:,1:T+1] + dRtdFtp * dFt[:,0:T])
    #gradient = (dStdA * dAdRt + dStdB*dBdRt) * (dRtdFt * dFt[:, 1:T + 1] + dRtdFtp * dFt[:, 0:T])
    gradient = (dRtdFt * dFt[:, 1:T + 1] + dRtdFtp * dFt[:, 0:T])
    return -1*np.sum(gradient,1)

def train(theta, X, Xn, T, M, mu, transactionCosts, startPos, rho, N):
    sharpe = 1
    previousSharpe = 0
    gradient = np.array([M + 2, T+1])

    #print (theta)
    for i in range(N):
        previousSharpe = sharpe;
        # Compute the Sharpe ratio #
        F = ComputeF(theta, Xn, T, M, startPos)  # Compute the values of vector of with current theta values.

        # print(F,F.size)
        Reward = RewardFunction(mu, F, transactionCosts, T, M, X)
        #print(Reward)
        sharpe = SharpeRatio(Reward)
        sharpe = -1 * sharpe;

        # Compute the gradient.
        gradient = GradientFunctionM(theta, X, Xn, T, M, mu, transactionCosts, startPos)
        theta = theta - rho*gradient
        #print (theta)

    return theta
#### END CORE FUNCTIONS ####


