import numpy as np
import math
from scipy.optimize import fmin_ncg
from RRL.UtilityFunctions import GetReturns
from RRL.UtilityFunctions import FeatureNormalize
import matplotlib.pyplot as plt

class RRL:
    def __init__(self, initial_theta, T, M, data, optimization_function="sharpe_ratio", rho=0.001, mu=1):
        # Set parameters for the rrl class.
        self.theta = initial_theta
        self.optimization_function = optimization_function
        self.rho = rho
        self.mu = mu
        # Set the size of the array F as T+1 since we need to initialize t-1 when t = 0.
        self.F = np.zeros(T + 1)
        self.R = np.zeros(T)

        self.data = data
        self.T = T
        self.M = M
        self.start_position = M
        self.finish_position = M + T

        self.r = GetReturns(data)
        self.rn = FeatureNormalize(self.r)

        self.update_theta()


        #self.theta = fmin_ncg(self.__ObjectiveFunction__, self.theta, self.__GradientFunctionM__,
        #                  args=(self.X, self.Xn, training_window, inputs_window, startPos), avextol=1e-8, maxiter=50)

    def reward_function(self):
        # Generate discrete decisions.
        # F = np.ceil(np.absolute(F)) * np.sign(F)
        # Return the rewards discounting transaction costs.
        self.R = self.mu * (self.F[0:self.T] * self.r[self.M:self.M + self.T] - self.T *
                                                np.abs(self.F[1:self.T + 1] - self.F[0:self.T]))

    @staticmethod
    def sharpe_ratio(reward):
        # Set the Reward signal input as an numpy array.
        reward = np.asarray(reward)
        # Get the average of the Rewards.
        A = reward.mean()
        # Get the variance of the rewards signal.
        B = np.square(reward).mean()
        # Compute and return the sharpe ratio of the array.
        return A / math.sqrt(B - math.pow(A, 2))

    def __compute_F__(self, theta=None):
        if theta is not None:
            self.theta = theta
        # Iterate over the window of size T.
        for i in range(1, self.T + 1):
            # Get the correspondent inputs of the neuron as a window of size M.
            data_inputs = self.r[(i - 1) + self.start_position - self.M: self.start_position + i - 1]
            # Create the neuron input by adding the bias and the last decision input.
            neuron_inputs = np.concatenate(([1], data_inputs, [self.F[i - 1]]), axis=0)
            # Compute the output using the activation function.
            self.F[i] = math.tanh(np.dot(self.theta, neuron_inputs))

    def __ObjectiveFunction__(self, theta=None):
        if theta is not None:
            self.theta = theta
        print(self.theta)
        # Compute the neuron output with the current theta value.
        self.__compute_F__()
        # Get the rewards according to the decisions in F.
        self.reward_function()
        # Check the type of function to be used for optimization.
        # Compute and return the Sharpe ratio #
        if self.optimization_function == "sharpe_ratio":
            # Negate the sharpe ratio since we want to maximize.
            sharpe = -1 * self.sharpe_ratio(self.R)
            return sharpe
        # Compute the returns.
        if self.optimization_function == "return":
            # Negate the rewards since we want to maximize.
            A = -1 * self.R.mean()
            return A

    def __GradientFunctionM__(self, theta=None):
        if theta is not None:
            self.theta = theta
        # Compute the neuron output at each time step.
        self.__compute_F__()
        # Compute the rewards at each time step according to the decisions in F.
        self.reward_function()

        # Initialize dFt as a matrix of zeros. (The gradient has a component per input) M + 2
        # gradient inputs * T time steps.
        dFt = np.zeros([self.M + 2, self.T + 1])

        # Compute the derivative of F as a vector of M + 2 (dFt). Start computing from the first element on to the
        # last, assuming t-1 when t = 0 is 0.
        for i in range(1, self.T + 1):
            # Create the inputs window.
            inputsWindow = self.rn[(i - 1) + self.start_position - self.M: self.start_position + i - 1]
            # Create the neuron input by adding the bias and the last decision.
            neuronInput = np.concatenate(([1], inputsWindow, [self.F[i - 1]]), axis=0)
            dFt[:, i] = (1 - math.pow(math.tanh(np.dot(self.theta, neuronInput)), 2)) * (
            neuronInput + self.theta[self.M + 1] * dFt[:, i - 1])

        # Set the values of A and B from the gradient formula.
        A = self.R.mean()  # Compute the mean.
        B = np.square(self.R).mean()  # Compute the standard deviation.

        # Compute the under part of the derivatives of S (Sharpe Ratio) with respect to A and B.
        A2 = math.pow(A, 2)
        under = math.pow(B - A2, 3 / 2)

        # Compute the partial derivatives.
        dStdA = B / under
        dStdB = -A / 2 * under
        dAdRt = 1
        dBdRt = 2 * A
        dRtdFt = -self.mu * self.rho * np.sign(self.F[1:self.T + 1] - self.F[0:self.T])
        dRtdFtp = self.mu * self.r[self.M:self.M + self.T] + self.mu * self.rho * np.sign(self.F[1:self.T + 1] -
                  self.F[0:self.T])

        # Check the type of function to be used for gradient computation.
        # Compute gradient with respect to the Sharpe ratio.
        if self.optimization_function == "sharpeRatio":
            gradient = (dStdA * dAdRt + dStdB * dBdRt) * (dRtdFt * dFt[:, 1:self.T + 1] + dRtdFtp * dFt[:, 0:self.T])
            gradient = np.asarray(gradient)
            # Negate the output gradient since we want to maximize.
            return -1 * np.sum(gradient, 1)
        # Compute the returns.
        if self.optimization_function == "return":
            # Negate the rewards since we want to maximize.
            gradient = (dRtdFt * dFt[:, 1:self.T+ 1] + dRtdFtp * dFt[:, 0:self.T])
            gradient = np.asarray(gradient)
            # Negate the output gradient since we want to maximize.
            return -1 * np.sum(gradient, 1)

    def update_theta(self, theta=None):
        if theta is not None:
            self.theta = theta

        self.theta = fmin_ncg(self.__ObjectiveFunction__, self.theta, self.__GradientFunctionM__, avextol=1e-8,
                              maxiter=50)

    def plot_decision(self, inputs_window, training_window, theta=None):
        startPos = inputs_window
        finishPos = startPos + training_window
        # Compute the output F.
        print("Computing neuron outputs F... ")
        self.__compute_F__(theta, self.Xn, training_window, inputs_window, startPos)
        # Compute the rewards
        print("Computing reward function... ")
        rewards = self.reward_function(training_window, inputs_window, self.X)
        # Compute the cumulative reward.
        rewards = rewards + 1  # Add one to the rewards vector such that the reward does not vanish.
        rewards = np.asarray(rewards)
        for i in range(1, rewards.size):
            rewards[i] = rewards[i - 1] * rewards[i]

        returns = self.X[startPos:finishPos] + 1
        for i in range(1, returns.size):
            returns[i] = returns[i - 1] * returns[i]

        plt.subplot(3, 1, 1)
        plt.title("Returns")
        plt.plot(returns)
        plt.xlim([0, finishPos])
        plt.subplot(3, 1, 2)
        plt.title("Neuron Output")
        plt.plot([0, training_window], [0, 0], color='k', linestyle='-', linewidth=2)
        plt.plot(self.F[1:], color='r', linestyle='--')
        plt.ylim([-1, 1])
        plt.xlim([0, finishPos])
        plt.subplot(3, 1, 3)
        plt.title("Agent Rewards")
        plt.plot(rewards)
        plt.xlim([0, finishPos])
        plt.show()

        fig, ax1 = plt.subplots()
        t = np.arange(0.0, 150.0, 1)
        ax1.plot(self.F[1:], 'y')
        ax1.set_xlabel('Days')
        # Make the y-axis label and tick labels match the line color.
        ax1.set_ylabel('Neuron Output', color='b')
        for tl in ax1.get_yticklabels():
            tl.set_color('b')

        B = self.F[1:] > 0
        for i, b in enumerate(B):
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