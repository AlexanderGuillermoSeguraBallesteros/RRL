'''
from scipy.optimize import fmin_ncg
from scipy.optimize import fmin_bfgs
from pyalgotrade.tools import yahoofinance
from numpy import asarray
from numpy import diag
from numpy import zeros_like
from numpy import genfromtxt
import numpy as np
from CoreFunctions_v2 import RewardFunction as RFv2
from CoreFunctions_v2 import ComputeF as CFv2
from CoreFunctions_v2 import SharpeRatio as SRv2
from CoreFunctions_v2 import ObjectiveFunction as OFv2
from CoreFunctions import RewardFunction as RFv1
from CoreFunctions import ComputeF as CFv1
from CoreFunctions import SharpeRatio as SRv1
from CoreFunctions import  ObjectiveFunction as OFv1
from UtilityFunctions import GetReturns

def rosen(x):
    """The Rosenbrock function"""
    return sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)

def rosen_der(x):
    xm = x[1:-1]
    xm_m1 = x[:-2]
    xm_p1 = x[2:]
    der = zeros_like(x)
    der[1:-1] = 200*(xm-xm_m1**2) - 400*(xm_p1 - xm**2)*xm - 2*(1-xm)
    der[0] = -400*x[0]*(x[1]-x[0]**2) - 2*(1-x[0])
    der[-1] = 200*(x[-1]-x[-2]**2)
    return der

def rosen_hess(x):
    x = asarray(x)
    H = diag(-400*x[:-1],1) - diag(400*x[:-1],-1)
    diagonal = zeros_like(x)
    diagonal[0] = 1200*x[0]-400*x[1]+2
    diagonal[-1] = 200
    diagonal[1:-1] = 202 + 1200*x[1:-1]**2 - 400*x[2:]
    H = H + diag(diagonal)
    return H

#x0 = [1.3, 0.7, 0.8, 1.9, 1.2]
#xopt = fmin_bfgs(rosen, x0)
#xopt = fmin_ncg(rosen, x0, rosen_der, fhess=rosen_hess, avextol=1e-8)

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
startPos = M
finishPos = startPos + T
transactionCosts = 0.001
mu = 1
F = np.zeros(T + 1)
dF = np.zeros(T + 1)

# Generate the returns.
X = GetReturns(trainingData)

# Generate theta
theta = np.ones(M+2)*1


print ("Compare computeF functions:")
windowSize = T + M
Fv1 = CFv1(X[0:windowSize],theta,T,M)
#print ("V1 --> ", Fv1)
Fv2 = CFv2(theta, X, T, M, startPos)
#print ("V2 --> ", Fv2)
print ("ComputeF difference: ", sum(Fv1-Fv2))

print ("Compare rewards")
Rv1 = RFv1(mu,Fv1,transactionCosts,T,M,X[0:windowSize])
Rv2 = RFv2(mu, Fv2, transactionCosts, T, M, X)
#print ("V1 --> ", Rv1)
#print ("V2 --> ", Rv2)
print ("Rewards difference: ", sum(Rv1-Rv2))

print ("Compare sharpe")
Srv1 = SRv1(Rv1)
Srv2 = SRv2(Rv2)
print ("Sharpe difference: ", Srv1-Srv2)


print ("Compare ObjectiveFunction")
Ofv1 = OFv1(theta, X[0:windowSize], mu, transactionCosts, M, T)
Ofv2 = OFv2(theta, X, T, M, mu, transactionCosts, startPos)
print (Ofv1,Ofv2)
#print xopt
'''

import quandl
data = quandl.get("WIKI/AAPL", start_date = "2001-12-31", end_date="2005-12-31", returns="numpy")
dataframe = quandl.get("WIKI/AAPL", start_date = "2001-12-31", end_date="2005-12-31")
print (dataframe["Adj. Close"])
print (data["Adj. Close"])