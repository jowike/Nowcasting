from scipy.signal import lfilter
from scipy.interpolate import splrep, splev
import numpy as np

def filter(x, k):
    """Apply a moving average filter with a window size of 2*k+1."""
    numerator = np.ones(2*k+1) / (2*k+1)
    return lfilter(numerator, [1], x)

def remNaNs_spline(X,options):
    """
    Treats NaNs in the dataset for use in Dynamic Factor Models (DFM).

    This function processes NaNs in a data matrix `X` according to five cases, 
    which are useful for running functions in the `DFM.m` file that do not 
    accept missing value inputs.

    Replication files for: 
    "Nowcasting", 2010, by Marta Banbura, Domenico Giannone, and Lucrezia Reichlin, 
    in Michael P. Clements and David F. Hendry, editors, Oxford Handbook on Economic Forecasting.

    The software can be freely used in applications. Users are kindly requested to 
    add acknowledgments to published work and cite the above reference in any resulting publications.

    Args:
        X (ndarray): Input data matrix of shape (T, n) where `T` is time and `n` is the number of series.
        options (dict): A dictionary with the following keys:
            - method (int): Determines the method for handling NaNs.
                - 1: Replaces all missing values using a filter.
                - 2: Replaces missing values after removing trailing and leading zeros 
                     (a row is 'missing' if more than 80% is NaN).
                - 3: Only removes rows with leading and closing zeros.
                - 4: Replaces missing values after removing trailing and leading zeros 
                     (a row is 'missing' if all are NaN).
                - 5: Replaces missing values using a spline and then applies a filter.
            - k (int): Used in MATLAB's filter function for the 1-D filter. 
              Controls the rational transfer function's numerator, where the 
              denominator is set to 1. The numerator takes the form 
              `ones(2*k+1, 1) / (2*k+1)`. See MATLAB's documentation for `filter()` for details.

    Returns:
        tuple:
            - X (ndarray): The processed data matrix.
            - indNaN (ndarray): A matrix indicating the location of missing values (1 for NaN).
    """
    T, N = X.shape  # Gives dimensions for data input
    k = options["k"]  # Inputted options
    method = options["method"]  # Inputted options
    indNaN = np.isnan(X)  # Returns location of NaNs
    if method == 1:   # replace all the missing values
        for i in range(N):  # loop through columns
            x = X[:, i]
            isnanx = indNaN[:, i]
            x[isnanx]  = np.nanmedian(x)  # Replace missing values series median
            x_MA = filter(np.concatenate(([x[0]] * k, x, [x[-1]] * k)), k)  # Apply filter
            x_MA = x_MA[2*k:]  # Match dimensions
            x[isnanx] = x_MA[isnanx]  # Replace missing observations with filtered values
            X[:, i] = x  # Replace vector
    elif method == 2:   # replace missing values after removing leading and closing zeros
        rem1 = np.sum(indNaN, axis=1) > N * 0.8  # Returns row sum for NaN values. Marks true for rows with more than 80% NaN
        nanLead = np.cumsum(rem1) == np.arange(1, T+1)
        nanEnd = np.cumsum(rem1[::-1]) == np.arange(1, T+1)
        nanEnd = nanEnd[::-1]  # Reverses nanEnd
        nanLE = nanLead | nanEnd

        X = X[~nanLE, :]  # Remove leading and trailing NaN rows
        indNaN = np.isnan(X)  # Index for missing values

        # Loop for each series
        for i in range(N):
            x = X[:, i]
            isnanx = np.isnan(x)
            t1 = np.where(~isnanx)[0][0]  # First non-NaN entry
            t2 = np.where(~isnanx)[0][-1]  # Last non-NaN entry

            # Interpolates without NaN entries in beginning and end
            tck = splrep(np.where(~isnanx)[0], x[~isnanx], s=0)
            x[t1:t2+1] = splev(np.arange(t1, t2+1), tck)
            isnanx = np.isnan(x)

            x[isnanx] = np.nanmedian(x)  # Replace NaNs with the median

            # Apply filter
            x_MA = filter(np.concatenate(([x[0]] * k, x, [x[-1]] * k)), k)
            x_MA = x_MA[2*k:]
            x[isnanx] = x_MA[isnanx]
            X[:, i] = x

    elif method == 3:  # Only remove rows with leading and closing zeros
        rem1 = np.sum(indNaN, axis=1) == N
        nanLead = np.cumsum(rem1) == np.arange(1, T+1)
        nanEnd = np.cumsum(rem1[::-1]) == np.arange(1, T+1)
        nanEnd = nanEnd[::-1]
        nanLE = nanLead | nanEnd

        # Remove leading and trailing NaN rows
        X = X[~nanLE, :]
        indNaN = np.isnan(X)

    elif method == 4:  # Remove rows with leading and closing zeros & replace missing values
        rem1 = np.sum(indNaN, axis=1) == N
        nanLead = np.cumsum(rem1) == np.arange(1, T+1)
        nanEnd = np.cumsum(rem1[::-1]) == np.arange(1, T+1)
        nanEnd = nanEnd[::-1]
        nanLE = nanLead | nanEnd

        # Remove leading and trailing NaN rows
        X = X[~nanLE, :]
        indNaN = np.isnan(X)

        for i in range(N):
            x = X[:, i]
            isnanx = np.isnan(x)
            t1 = np.where(~isnanx)[0][0]
            t2 = np.where(~isnanx)[0][-1]

            # Interpolation
            tck = splrep(np.where(~isnanx)[0], x[~isnanx], s=0)
            x[t1:t2+1] = splev(np.arange(t1, t2+1), tck)
            isnanx = np.isnan(x)

            x[isnanx] = np.nanmedian(x)  # Replace NaNs with the median
            
            # Apply filter
            x_MA = filter(np.concatenate(([x[0]] * k, x, [x[-1]] * k)), k)
            x_MA = x_MA[2*k:]
            x[isnanx] = x_MA[isnanx]
            X[:, i] = x

    elif method == 5:  # Replace missing values
        indNaN = np.isnan(X)
        for i in range(N):
            x = X[:, i]
            isnanx = np.isnan(x)
            t1 = np.where(~isnanx)[0][0]
            t2 = np.where(~isnanx)[0][-1]

            # Interpolation
            tck = splrep(np.where(~isnanx)[0], x[~isnanx], s=0)
            x[t1:t2+1] = splev(np.arange(t1, t2+1), tck)
            isnanx = np.isnan(x)

            x[isnanx] = np.nanmedian(x)  # Replace NaNs with the median

            # Apply filter
            x_MA = filter(np.concatenate(([x[0]] * k, x, [x[-1]] * k)), k)
            x_MA = x_MA[2*k:]
            x[isnanx] = x_MA[isnanx]
            X[:, i] = x

    return X, indNaN
