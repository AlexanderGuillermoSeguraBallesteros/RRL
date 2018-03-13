
'''
This file contains the core functions for the basic RRL algorithm. Two measures of performance are applied to the
algorithm: Returns and Sharpe Ratio. The reward function is implemented. considering transaction costs.
'''
import numpy as np
import math

## REWARD FUNCTION ##
'''
Function to get the value of the rewards for an array of size T from the returns.
F -> Vector containing the discrete output Ft at each time t from 1 to T + 1.
X -> Vector containing the returns for the whole series.
mu -> Number of shares.
T -> Data window to train over.
transactionCosts -> percentage of the return to discount from each transaction.
M -> Number of inputs to the neuron.
'''
def RewardFunction (mu, F, transactionCosts, T, M, X):
    #F = np.ceil(np.absolute(F)) * np.sign(F) # Generate discrete decisions.
    return mu*(F[0:T]*X[M:M+T] - transactionCosts*np.abs(F[1:T+1]-F[0:T]))  # Return the rewards discounting transaction
                                                                            # costs.
## END REWARD FUNCTION ##

## SHARPE RATIO ##
'''
Compute of the sharpe ratio function for a time series input.
'''
def SharpeRatio(Reward):
    # Set the Reward signal input as an numpy array.
    Reward = np.asarray(Reward)
    # Get the average of the Rewards.
    A = Reward.mean()
    # Get the variance of the rewards signal.
    B = np.square(Reward).mean()
    # Compute and return the sharpe ratio of the array.
    return A/math.sqrt(B-math.pow(A, 2))
## END SHARPE RATIO ##


'''
Function to compute the value of the neuron output at each time step t from an array of size T.
'''
## COMPUTE F ##
# Returns all the values of F from time 0 to T+1.
def ComputeF(theta, X, T, M, startPos):
    # Set the size of the array F as T+1 since we need to initialize t-1 when t = 0.
    F = np.zeros(T+1)
    # Iterate over the window of size T.
    for i in range(1, T+1):
        # Get the correspondent inputs of the neuron as a window of size M.
        inputsWindow = X[(i-1) + startPos - M: startPos + i - 1]
        # Create the neuron input by adding the bias and the last decision input.
        neuronInput = np.concatenate(([1], inputsWindow, [F[i-1]]), axis=0)
        # Compute the output using the activation function.
        F[i] = math.tanh(np.dot(theta, neuronInput))

    return F
## END COMPUTE F ##

'''
The function to be optimized with respect to the parameters theta.
'''
## OBJECTIVE FUNCTION ##
def ObjectiveFunction(theta, X, Xn, T, M, mu, transactionCost, startPos, functionName):
    # Compute the neuron output with the current theta value.
    F = ComputeF(theta, Xn, T, M, startPos)
    # Get the rewards according to the decisions in F.
    Reward = RewardFunction(mu, F, transactionCost, T, M, X)
    # Check the type of function to be used for optimization.
    # Compute and return the Sharpe ratio #
    if functionName == "sharpeRatio":
        # Negate the sharpe ratio since we want to maximize.
        sharpe = -1*SharpeRatio(Reward)
        return sharpe
    # Compute the returns.
    if functionName == "return":
        # Negate the rewards since we want to maximize.
        A = -1*Reward.mean()
        return A
## END OBJECTIVE FUNCTION ##

'''
Compute the gradient of the objective function. The objective function is derived from the sharpe ratio formula
using the chain rule.
'''
## GRADIENT ##
def GradientFunctionM(theta, X, Xn, T, M, mu, transactionCosts, startPos, functionName):
    # Compute the neuron output at each time step.
    F = ComputeF(theta, Xn, T, M, startPos)
    # Compute the rewards at each time step according to the decisions in F.
    rewards = RewardFunction(mu, F, transactionCosts, T, M, X)

    # Initialize dFt as a matrix of zeros. (The gradient has a component per input) M + 2
    # gradient inputs * T time steps.
    dFt = np.zeros([M+2, T + 1])

    # Compute the derivative of F as a vector of M + 2 (dFt). Start computing from the first element on to the
    # last, assuming t-1 when t = 0 is 0.
    for i in range(1, T+1):
        # Create the inputs window.
        inputsWindow = Xn[(i - 1) + startPos - M: startPos + i - 1]
        # Create the neuron input by adding the bias and the last decision.
        neuronInput = np.concatenate(([1], inputsWindow, [F[i - 1]]), axis=0)
        dFt[:, i] = (1 - math.pow(math.tanh(np.dot(theta, neuronInput)), 2))*(neuronInput + theta[M+1]*dFt[:, i-1])

    # Set the values of A and B from the gradient formula.
    A = rewards.mean()  # Compute the mean.
    B = np.square(rewards).mean()  # Compute the standard deviation.

    # Compute the under part of the derivatives of S (Sharpe Ratio) with respect to A and B.
    A2 = math.pow(A, 2)
    under = math.pow(B-A2, 3/2)

    # Compute the partial derivatives.
    dStdA = B / under
    dStdB = -A / 2*under
    dAdRt = 1
    dBdRt = 2*A
    dRtdFt = -mu * transactionCosts * np.sign(F[1:T+1] - F[0:T])
    dRtdFtp = mu * X[M:M+T] + mu * transactionCosts * np.sign(F[1:T+1] - F[0:T])

    # Check the type of function to be used for gradient computation.
    # Compute gradient with respect to the Sharpe ratio.
    if functionName == "sharpeRatio":
        gradient = (dStdA * dAdRt + dStdB * dBdRt) * (dRtdFt * dFt[:, 1:T + 1] + dRtdFtp * dFt[:, 0:T])
        # Negate the output gradient since we want to maximize.
        return -1 * np.sum(gradient, 1)
    # Compute the returns.
    if functionName == "return":
        # Negate the rewards since we want to maximize.
        gradient = (dRtdFt * dFt[:, 1:T + 1] + dRtdFtp * dFt[:, 0:T])
        # Negate the output gradient since we want to maximize.
        return -1 * np.sum(gradient, 1)
## END GRADIENT ##

## TRAINING ##
def train(theta, X, Xn, T, M, mu, transactionCosts, startPos, rho, N, functionName):
    sharpe = 1
    previousSharpe = 0
    gradient = np.array([M + 2, T+1])

    for i in range(N):
        #previousSharpe = sharpe;

        # Compute the Sharpe ratio #
        #F = ComputeF(theta, Xn, T, M, startPos)  # Compute the values of vector of with current theta values.

        #Reward = RewardFunction(mu, F, transactionCosts, T, M, X)
        #sharpe = SharpeRatio(Reward)
        #sharpe = -1 * sharpe;

        # Compute the gradient.
        gradient = GradientFunctionM(theta, X, Xn, T, M, mu, transactionCosts, startPos, functionName)
        # Weights update rule.
        theta = theta + rho*gradient
    return theta
## END TRAINING ##

