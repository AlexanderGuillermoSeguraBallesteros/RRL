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


# Program parameters
transactionCost = 0.001 # Set the transaction cost as a percentage of the transaction.
T = 100
M = 10

#### CORE FUNCTIONS ####

## REWARD FUNCTION ##
# Function to get the transaction cost at a time step t.
# F -> Vector containing the discrete output Ft at each time t from 1 to T + 1.
# X -> Vector containing the returns for the whole series.
def RewardFunction (mu, F, transactionCosts, T, M, X):
    #return mu*(F[0:T]*X[M:M+T-1] - transactionCosts*np.sign(F[1:T-1]-F[0:T-2]))
    return mu*(F[0:T]*X[M:M+T]- transactionCosts*np.sign(F[1:T+1]-F[0:T]))
## END REWARD FUNCTION ##

## SHARPE RATIO ##
def SharpeRatio(Reward):
    Reward = np.asarray(Reward)
    A = Reward.mean()
    B = Reward.std()
    return A/math.sqrt(B-math.pow(A, 2))
## END SHARPE RATIO ##


## COMPUTE F ##
# Returns all the values of F from time 0 to T+1.
def ComputeF(X, theta, T, M):
    #print(X)
    F = np.zeros(T+1)
    for i in range(1, T+1):
        Xt = np.concatenate(([1], X[i-1:i+M-1], [F[i-1]]), axis=0)
        F[i] = math.tanh(np.dot(theta, Xt))

    return F
## END COMPUTE F ##


## UTILITY FUNCTION ##
def ObjectiveFunction(theta, X, mu, transactionCost, M, T):


    # Compute the Sharpe ratio #
    F = ComputeF(X, theta, T, M) # Compute the values of vector of with current theta values.
    #print(F,F.size)
    Reward = RewardFunction(mu, F, transactionCost, T, M, X)
    sharpe = SharpeRatio(Reward)
    sharpe = -1*sharpe;


    # Compute gradient #
    # Compute dF / dtheta
    '''
    dFt = np.zeros((M + 2, T + 1), dtype='double')
    for i in range(1, T + 1):
        Xt = [1, X[i - 1:i + M - 2], F[i - 1]]
        dFt[:, i] = (1 - math.tanh(theta * Xt) ^ 2) * (Xt + theta[M+2-1, 1]*dFt[:, i-1])
    '''
    # Compute gradient

    return sharpe
## END UTILITY FUNCTION ##

## GRADIENT ##
def gradientFunctionM(M,F):
    1

#### END CORE FUNCTIONS ####


