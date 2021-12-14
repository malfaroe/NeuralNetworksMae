#Methods for data analysis

##Import libraries 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime, timedelta

# import all libraries and dependencies for data visualization
pd.options.display.float_format='{:.4f}'.format
plt.rcParams['figure.figsize'] = [8,8]
pd.set_option('display.max_columns', 350)
pd.set_option('display.max_colwidth', -1) 
pd.set_option("display.max_rows", 500)
sns.set(style='darkgrid')
import matplotlib.ticker as ticker
import matplotlib.ticker as plticker


# import all libraries and dependencies for machine learning
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.base import TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import RobustScaler 
from sklearn.preprocessing import StandardScaler


from sklearn.compose import make_column_transformer


import statsmodels.api as sm
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LinearRegression, OrthogonalMatchingPursuit, Lasso, LassoLarsIC, ElasticNet, ElasticNetCV
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import f_regression, mutual_info_regression, SelectKBest, RFECV, SelectFromModel
from sklearn.model_selection import train_test_split

from scipy import stats
from scipy.stats import norm, kurtosis, skew

# Import specific libraries
import statsmodels.api as sm
from statsmodels.stats import diagnostic as diag
from statsmodels.stats.outliers_influence import variance_inflation_factor

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import RobustScaler, PolynomialFeatures, StandardScaler, LabelEncoder
from sklearn.feature_selection import f_regression, mutual_info_regression, SelectKBest, RFECV, SelectFromModel
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
from sklearn.feature_extraction import FeatureHasher
from sklearn.decomposition import PCA, KernelPCA
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LinearRegression, OrthogonalMatchingPursuit, Lasso, LassoLarsIC, ElasticNet, ElasticNetCV
from sklearn.linear_model import SGDRegressor, PassiveAggressiveRegressor, HuberRegressor, BayesianRidge
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, BaggingRegressor, ExtraTreesRegressor
# import xgboost as xgb
# from xgboost import XGBRegressor, plot_importance
import lightgbm as lgb

from patsy import dmatrices

from sklearn.model_selection import GridSearchCV, cross_val_score, KFold, cross_val_predict, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Stats
from scipy.stats import skew, norm
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax

from scipy import stats
from scipy.stats import skew, norm, probplot, boxcox
from scipy.special import boxcox1p
from patsy import dmatrices

pd.set_option('expand_frame_repr', False)



#Modulo: General summary of dataset

def data_summary(df, target):
    """General summary of dataset"""
    num = df.select_dtypes(exclude= object).columns
    cols = ["Dtype","Uniques", "Nulls","% Nulls", "Skew", "Kurtosis", "Correlation"]
    ind = df.columns
    dict = {}
    dict["Dtype"] = [df[i].dtype for i in ind]
    dict["Uniques"] = [df[i].nunique() for i in ind]
    dict["Nulls"] =[df[i].isnull().sum().sum() for i in ind]
    dict["% Nulls"] = np.round([df[i].isnull().sum().sum()/len(df[i]) for i in ind], decimals = 2)
    dict["Skew"] =  np.round(df.skew(), decimals = 3) 
    dict["Kurtosis"] = np.round(df.kurt(), decimals = 3)
    dict["Correlation"] = np.round(df.corr()[target], decimals = 3)
    summary = pd.DataFrame(dict, columns = cols, index = ind).sort_values(by = "Dtype").sort_values(by = "Correlation", ascending = False)
    summary = summary.iloc[summary.Correlation.abs().argsort()][::-1]
    return summary



def features_profile(data, target):
    """GENERATE GROUPS OF FEATURES ACCORDING WITH THEIR SKEW/KURT/CORR VALUES"""
    #Creates a summary first
    sum = data_summary(data, target).sort_values(by = "Dtype").sort_values(by = "Correlation", ascending = False)

    ### Features high skewed right, heavy-tailed distribution, and with high correlation: apply transformations and manage outliers
    sum_1 = sum[(abs(sum["Skew"]) > 1) & (abs(sum["Kurtosis"]) > 3) & (abs(sum["Correlation"]) > 0.5)][["Skew", "Kurtosis", "Correlation"]]
    print("1. Features highly skewed right, heavy-tailed distribution, and with high correlation:")
    print("What to do: apply transformations and manage outliers")
    print("")
    print(sum_1)
    print("")

    ### Features skewed, heavy-tailed distribution, and with good correlation: apply transformations and manage outliers

    sum_2 = sum[(abs(sum["Skew"] > 1)) & (abs(sum["Kurtosis"]) > 1) & (abs(sum["Correlation"]) > 0.05)][["Skew", "Kurtosis", "Correlation"]].drop(sum_1.index)
    print("2. Features skewed, heavy-tailed distribution, and with high correlation")
    print("What to do: apply transformations and manage outliers")
    print("")
    print(sum_2)
    print("")

    ##Features high skewed, heavy-tailed distribution, and with low correlation: Maybe we can drop these features, or just use they with other to create a new more important features:

    sum_3 = sum[(abs(sum["Skew"] > 1)) & (abs(sum["Kurtosis"]) > 1) & (abs(sum["Correlation"]) > 0.01)][["Skew", "Kurtosis", "Correlation"]].drop((sum_1 + sum_2).index)
    print("3. Features skewed, heavy-tailed distribution, and with low correlation")
    print("What to do: Maybe we can drop these features, or just use they with other to create a new more important features")
    print("")
    print(sum_3)



#Method for fast plotting

def plot_dist(feat, target):
    fig = plt.figure(figsize=(17,5))
    ax = fig.add_subplot(121)
    sns.scatterplot(x =train[feat], y = train[target], ax = ax)




def plot_feats (df, cols,target, hue):
    """method for plotting relationship of target with high correlated variables
    using or not a hue
    cols = list of features to plot"""
    
    if hue in cols:
        cols.remove(hue)
    if target in cols:
        cols.remove(target)
    sns.reset_defaults()
    sns.set(style="ticks", color_codes=True)
#     fig = plt.figure(figsize = (15,10))
    sns.set(font_scale= 1.0)
    if hue == None:
        fig_s = plt.figure(figsize = (15,25))
        for i, c in enumerate(cols):
            fig_i= fig_s.add_subplot(420 + i + 1)
            sns.scatterplot(x = df[c], y = df[target], palette= 'Spectral')
            plt.show()
    else:
        #  Box plot hue/target
        fig =  plt.figure(figsize = (15,10))
        fig_1 = fig.add_subplot(221)
        sns.boxplot(x=hue, y=target, data=df[[target, hue]])
        plt.show()

        fig_s = plt.figure(figsize = (15,25))
        for i, c in enumerate(cols):
            fig_i= fig_s.add_subplot(420 + i + 1)
            sns.scatterplot(x = df[c], y = df[target], hue=df[hue], palette= 'Spectral')
            plt.show()
   
      







