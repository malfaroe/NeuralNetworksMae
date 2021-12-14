#Testing a massive feature generator both for num and cat

import pandas as pd 
import numpy as np 
import config

from itertools import combinations 
from sklearn.preprocessing import PolynomialFeatures


#Input data structure: train+test+target
#Processing all but target

df = pd.read_csv(config.INPUT_FILE)
#Selecciona todas las cols excepto target
df_sel = df.loc[:, df.columns != config.TARGET]

#Splitting cat and nums
cat_feats = df_sel.select_dtypes(include= object).columns
num_feats = df_sel.select_dtypes(exclude = object).columns


print("Initial features:", df_sel.shape[1])
print("Cat feats:", cat_feats)
print("Num feats:", num_feats)

#########
# # Categorical processing
# # Crear combinaciones de categoricals
# pairs = list(combinations(cat_feats, 2))
# for pair in pairs:
#     df_sel[pair[0] + "_" + pair[1]] = df_sel[pair[0]].astype(str)+ "_" 
#     + df_sel[pair[1]].astype(str)


# print("Total new features after cat combinations:", df_sel.shape[1])




# # #PART 2: #Create new numerical feats: binning, polynomial feats

# # #binning


# # Polynomial regressor of order 3 with ConstructArea (1,a,a2,a3)
# print("Polynomial regressor of order 2:")
# poly_2 = PolynomialFeatures(degree=2, interaction_only=False,
# include_bias=False) #instanciamos
# poly = poly_2.fit_transform(df_sel[num_feats]) # se crean todos los features nuevo
# col_poly = ["poly" + str(i) for i in range(poly.shape[1])]
# df_poly = pd.DataFrame(poly, columns = col_poly)

# print("Total num features after 2nd order poly creation:", df_poly.shape[1])
# #Get back together with target
# df = pd.concat((df[config.TARGET], df_sel, df_poly), axis = 1)

#######

#Save the data with new features
df.to_csv("../input/data_feat_gen.csv", index  = False)
print("Final features:", df.shape[1])
print("Final columns:", df.columns)
print(df.head(5))

