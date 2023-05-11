import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
import numpy as np
from env import user,password,host
import os
import wrangle
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

import sklearn.preprocessing
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

def plot_variable_pairs(df):
    '''
    accepts a dataframe as input and plots all of the pairwise relationships along with the regression line for each pair
    pay attention to df size, may want to use a sample to speed things up
    '''
    sns.set(style="ticks", color_codes=True)
    sns.pairplot(df, kind='reg')
    
    
def plot_categorical_and_continuous_vars(df, cat_var, cont_var,hue_var=None):
    '''
    accepts your dataframe and the name of the columns that hold the continuous and categorical features and outputs 3 different plots for visualizing a categorical variable and a continuous variable
    '''
    # plot 1: boxplot
    fig, axs = plt.subplots(figsize=(10,6))
    sns.boxplot(x=cat_var, y=cont_var, hue=hue_var, data=df)
    axs.set_title(f"{cat_var} vs {cont_var}")
    axs.set_xlabel(cat_var)
    axs.set_ylabel(cont_var)
    plt.show()
    
    # plot 2: violinplot
    fig, axs = plt.subplots(figsize=(10,6))
    sns.violinplot(x=cat_var, y=cont_var, hue=hue_var, data=df)
    axs.set_title(f"{cat_var} vs {cont_var}")
    axs.set_xlabel(cat_var)
    axs.set_ylabel(cont_var)
    plt.show()
    
    # plot 3: swarmplot
    fig, axs = plt.subplots(figsize=(10,6))
    sns.swarmplot(x=cat_var, y=cont_var, hue=hue_var, data=df)
    axs.set_title(f"{cat_var} vs {cont_var}")
    axs.set_xlabel(cat_var)
    axs.set_ylabel(cont_var)
    plt.show()

    
    #################### Evaluate######################
    
def plot_residuals(y, yhat):
    '''
    plot_residuals(y, yhat): creates a residual plot
    '''
    residuals = y - yhat
    plt.scatter(yhat, residuals)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted values')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.show()

def regression_errors(y, yhat):
    '''
    regression_errors(y, yhat): returns the following values:
    sum of squared errors (SSE)
    explained sum of squares (ESS)
    total sum of squares (TSS)
    mean squared error (MSE)
    root mean squared error (RMSE)
    '''
    SSE = np.sum((y - yhat) ** 2)
    ESS = np.sum((yhat - np.mean(y)) ** 2)
    TSS = np.sum((y - np.mean(y)) ** 2)
    MSE = SSE / len(y)
    RMSE = np.sqrt(MSE)
    return SSE, ESS, TSS, MSE, RMSE

def baseline_mean_errors(y):
    '''
    baseline_mean_errors(y): computes the SSE, MSE, and RMSE for the baseline model
    '''
    baseline_prediction = np.mean(y)
    SSE = np.sum((y - baseline_prediction) ** 2)
    MSE = SSE / len(y)
    RMSE = np.sqrt(MSE)
    return SSE, MSE, RMSE

def better_than_baseline(y, yhat):
    '''
    better_than_baseline(y, yhat): returns true if your model performs better than the baseline, otherwise false
    '''
    SSE_model, _, _, _, _ = regression_errors(y, yhat)
    SSE_baseline, _, _ = baseline_mean_errors(y)
    return SSE_model < SSE_baseline

def evaluate(y,yhat):
    '''
    all in one funct to return:
    plot_residuals(y, yhat): creates a residual plot
    regression_errors(y, yhat): returns the following values:
    sum of squared errors (SSE)
    explained sum of squares (ESS)
    total sum of squares (TSS)
    mean squared error (MSE)
    root mean squared error (RMSE)
    baseline_mean_errors(y): computes the SSE, MSE, and RMSE for the baseline model
    better_than_baseline(y, yhat): returns true if your model performs better than the baseline, otherwise false
    '''
    # plot residuals
    plot_residuals(y, yhat)
    
    SSE, ESS, TSS, MSE, RMSE = regression_errors(y, yhat)
    print("SSE:", SSE)
    print("ESS:", ESS)
    print("TSS:", TSS)
    print("MSE:", MSE)
    print("RMSE:", RMSE)
    
    SSE_baseline, MSE_baseline, RMSE_baseline = baseline_mean_errors(y)
    print("Baseline SSE:", SSE_baseline)
    print("Baseline MSE:", MSE_baseline)
    print("Baseline RMSE:", RMSE_baseline)
    
    if better_than_baseline(y, yhat):
        print("The model performs better than the baseline.")
    else:
        print("The model does not perform better than the baseline.")