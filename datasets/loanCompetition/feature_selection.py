import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor, KNeighborsClassifier
import plotly.graph_objs as go
import plotly.offline as py
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, RandomizedSearchCV, RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, roc_auc_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from lightgbm import LGBMClassifier
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier


from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score 


from sklearn.feature_selection import VarianceThreshold

import config

from patsy import dmatrices


# Import specific libraries
import statsmodels.api as sm
from statsmodels.stats import diagnostic as diag
from statsmodels.stats.outliers_influence import variance_inflation_factor



class Selector:
    def __init__(self):
        # self.target = target
        # self.df = df
        pass
        
    # def rescale(self, df, target):
    #     #Rescale data if necessary
    #     X2 = df.copy()
    #     y2 = X2.pop(target)
    #     scaler = StandardScaler().fit(X2)
    #     XRescaled = scaler.transform(X2)
    #     X_rescaled = pd.DataFrame(XRescaled, columns = X2.columns)
    #     print("1. Data rescaling:")
    #     print("Data has been rescaled")
    #     return pd.concat((y2, X_rescaled), axis = 1)

    def variance_selector(self, df, target):
        #Rescale data if necessary
        print("1.Variance Threshold feature selection:")
        print("")
        X2 = df.copy()
        y2 = X2.pop(target)
        print("Initial features:", X2.shape[1])
        low = [col for col in X2.columns if X2[col].std() < 0.5]
        # print("Estos son antes de escale:", low)
        # scaler = StandardScaler().fit(X2)
        # XRescaled = scaler.transform(X2)
        # X_rescaled = pd.DataFrame(XRescaled, columns = X2.columns)
        # low = [col for col in X2.columns if X[col].std() < 0.5]
        #Analysis of amount of variation and droping all features with low variance
        var_tresh = VarianceThreshold(threshold = 0.5)
        var_tresh.fit_transform(X2)
        data_transformed = X2.loc[:, var_tresh.get_support()]
        #print("Selected feat:", data_transformed.columns)
        print("Removed features:", set(X2.columns) - set(data_transformed.columns))
        print("Final features:", data_transformed.shape[1])
        print("{} features with low variance removed".format(X2.shape[1] - data_transformed.shape[1]))
        #Rejoin
        df = pd.concat((y2, data_transformed), axis = 1)
        return df


    def single_value_dominate(self, df, target):
        print("2.Removing columns with single value dominating > 95%:")
        print("")
        X2 = df.copy()
        y2 = X2.pop(target)
        remove_cols = []
        for col in X2.columns:
            count= 0
            count = sum([+1 for i in X2[col].values if i == X2[col].mode()[0]])
            if count/X2[col].shape[0] >= 0.95:
                remove_cols.append(col)
        selected_cols = set(df.columns) - set(remove_cols)
        print("Removed cols:", len(remove_cols))
        print("Total selected cols:", len(selected_cols))
        df = df[selected_cols]
        return df


    def calcDrop(self, res):
        # All variables with correlation > cutoff
        all_corr_vars = list(set(res['v1'].tolist() + res['v2'].tolist()))
        
        # All unique variables in drop column
        poss_drop = list(set(res['drop'].tolist()))

        # Keep any variable not in drop column
        keep = list(set(all_corr_vars).difference(set(poss_drop)))
        
        # Drop any variables in same row as a keep variable
        p = res[ res['v1'].isin(keep)  | res['v2'].isin(keep) ][['v1', 'v2']]
        q = list(set(p['v1'].tolist() + p['v2'].tolist()))
        drop = (list(set(q).difference(set(keep))))

        # Remove drop variables from possible drop 
        poss_drop = list(set(poss_drop).difference(set(drop)))
        
        # subset res dataframe to include possible drop pairs
        m = res[ res['v1'].isin(poss_drop)  | res['v2'].isin(poss_drop) ][['v1', 'v2','drop']]
            
        # remove rows that are decided (drop), take set and add to drops
        more_drop = set(list(m[~m['v1'].isin(drop) & ~m['v2'].isin(drop)]['drop']))
        for item in more_drop:
            drop.append(item)
            
        return drop

    def corrX_new(self, df, target, cut) :
        print("")
        print("3. Removing features with high pairwise correlation")
        # Get correlation matrix and upper triagle
        df2 = df.drop(target, axis = 1)

        corr_mtx = df2.corr().abs()
        avg_corr = corr_mtx.mean(axis = 1)
        up = corr_mtx.where(np.triu(np.ones(corr_mtx.shape), k=1).astype(np.bool))

        dropcols = list()

        res = pd.DataFrame(columns=(['v1', 'v2', 'v1.target', 
                                        'v2.target','corr', 'drop' ]))

        for row in range(len(up)-1):
            col_idx = row + 1
            for col in range (col_idx, len(up)):
                if(corr_mtx.iloc[row, col] > cut):
                    if(avg_corr.iloc[row] > avg_corr.iloc[col]): 
                        dropcols.append(row)
                        drop = corr_mtx.columns[row]
                    else: 
                        dropcols.append(col)
                        drop = corr_mtx.columns[col]
                    
                    s = pd.Series([ corr_mtx.index[row],
                    up.columns[col],
                    avg_corr[row],
                    avg_corr[col],
                    up.iloc[row,col],
                    drop],
                    index = res.columns)
            
                    res = res.append(s, ignore_index = True)

        dropcols_names = self.calcDrop(res)
        print("{} features removed".format(len(dropcols_names)))
        print("Features removed:", dropcols_names)
        print("")
        print("Selected features:", df.shape[1] - len(dropcols_names))
        selected_cols = set(df.columns) - set(dropcols_names)
        print("")
        print("Selected:", selected_cols)
        return df[selected_cols]

    
    # def VIF(self, target_name, data, y): #for detecting multicollinearity
    #     """toma dataset y devuelven un dataframe 
    #     con los vifs ordenados"""
        
    #     # #Creamos el dataframe con data escalada que sirve de input para dmatrices
    #     # scale = StandardScaler(with_std= False) #we scale data previuously using z = (x - u) / s. with:_std es falso porque no es 1
    #     # df = pd.DataFrame(scale.fit_transform(data), columns = cols) #cols: todos los names of predictors
    #     # df["SalePrice"] = y.values
        
    #     #Hacemos una regression usando el sistema dmatrices, de donde obtenemos x e y
    #     cols = data.columns
    #     features = "+".join(cols) #crea un string de todos los nombres con un + entre cada uno
    #     y, X = dmatrices(target_name +  '~' + features, data = data, return_type= "dataframe")
        
    #     #Calculamos los VIF for each feature usando como input x,y
    #     vif = pd.DataFrame()
    #     vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    #     vif["Features"] = X.columns
        
    #     #Show
    #     display(vif.sort_values("VIF Factor"))
        
    #     return vif

    # def calc_vif(self, X):

    #     # Calculating VIF
    #     vif = pd.DataFrame()
    #     vif["variables"] = X.columns
    #     vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    #     return(vif)

    def calculate_vif_(self, X, thresh=5.0):
        cols = X.columns
        variables = np.arange(X.shape[1])
        dropped=True
        while dropped:
            dropped=False
            c = X[cols[variables]].values
            vif = [variance_inflation_factor(c, ix) for ix in np.arange(c.shape[1])]

            maxloc = vif.index(max(vif))
            if max(vif) > thresh:
                print('dropping \'' + X[cols[variables]].columns[maxloc] + '\' at index: ' + str(maxloc))
                variables = np.delete(variables, maxloc)
                dropped=True

        print('Remaining variables:')
        print(X.columns[variables])
        return X[cols[variables]]


    def corr_target(self, df):
        print("")
        print("Remove features with low target correlation:")
        #Removes all feats with correlation under threshold
        remove_features = [feat for feat in df.columns if df.corr().abs()[[config.TARGET]].loc[feat, :][0] < 0.05]
        selected_cols = set(df.columns) - set(remove_features)
        print("Total features removed:", len(remove_features))
        print("Final features selected:", len(selected_cols))

        return df[selected_cols]

if __name__ == "__main__":
    train =  pd.read_csv("../input/train_final.csv")


    ##########
    # slc = Selector()
    # # df_r = slc.rescale(df, target = "Survived")
    # df_v = slc.variance_selector(train, target = config.TARGET)
    # df_svd =slc.single_value_dominate(df_v, target = config.TARGET )
    # df_corr_f = slc.corrX_new(df_svd, target = config.TARGET, cut = 0.65)
    # #Aqui multicollinearity
    # # print("4. Multicollinearity analysis")
    # # df_multi = slc.calculate_vif_(df_corr_f, thresh=5.0)
    # #Correlation with target
    # df_corr_target = slc.corr_target(df_corr_f)

    # #Update train and test sets with selected features
    # train = train[df_corr_target.columns]

    #############

    if config.KAGGLE:
        X_test =  pd.read_csv("../input/test_final.csv")
        test_columns = train.loc[:, train.columns != config.TARGET].columns
        print("test columns:", test_columns)
        X_test = X_test[test_columns]
        print("Nans in test:", X_test.isnull().sum().sum())
        X_test.to_csv("../input/new_test_final.csv", index = False)
        train.to_csv("../input/new_train_final.csv", index = False)

    else:

        #Saving the updated 
        train.to_csv("../input/new_train_final.csv", index = False)





        