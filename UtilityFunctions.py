import numpy as np
import math
from scipy.optimize import fmin_bfgs
from numpy import asarray
from numpy import diag
from numpy import zeros_like


#### CORE FUNCTIONS ####

## GET RETURNS ##

def GetReturns(X):
    X = np.asarray(X)

    # Parameters
    arrayLength = X.size - 1 # Set the length of the array
    returns = np.zeros(arrayLength - 1) # Initialize the array to zeros.

    # Compute returns for each time step.
    for i in range(1, arrayLength):
        returns[i-1] = (X[i]-X[i-1])/X[i]

    return returns

## END GET RETURNS ##

## FEATURE NORMALIZE ##
def FeatureNormalize(X):
    X_norm = X
    mu = np.mean(X)
    sigma = np.std(X)
    d = mu/sigma
    X_norm = (X - mu)/sigma
    return  X_norm

## END FEATURE NORMALIZE ##

#### END CORE FUNCTIONS ####
