from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import accuracy_score, r2_score
from scipy.stats import pearsonr
import numpy as np

def RMSE(pred, true):
    # Calculate Root Mean Square Error
    return np.sqrt(MSE(true, pred))

def MAPE(pred, true):
    # Calculate Mean Absolute Percentage Error
    return np.mean(np.abs((true - pred) / true))

def MSPE(pred, true):
    # Calculate Mean Square Percentage Error
    return np.mean(np.square((true - pred) / true))

def RSE(pred, true):
    # Calculate Residual Sum of Squares
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))

def CORR(pred, true):
    # Calculate Pearson correlation coefficient
    corr, _ = pearsonr(true, pred)
    return corr

def ACCURACY(pred, true):
    # Calculate accuracy score
    return accuracy_score(true, pred)

def R2(pred, true):
    # Calculate coefficient of determination
    return r2_score(true, pred)

def metric(pred, true):
    mae = MAE(true, pred)
    mse = MSE(true, pred)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    rse = RSE(pred, true)
    corr = CORR(pred, true)
    acc = ACCURACY(pred, true)
    r2 = R2(pred, true)

    return mae, mse, rmse, mape, mspe, rse, corr, acc, r2
