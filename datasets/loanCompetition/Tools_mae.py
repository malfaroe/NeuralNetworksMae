#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


###BLOCK LIST COMPARE: MY FIRST TOOL

def data_compare(a,b, dataframe = True):
     ## CHEQUEAR QUE AMBOS COMPARANDOS SEAN LISTAS O DF 
        
    if ((isinstance(a, pd.DataFrame) & isinstance(b, list)) or (isinstance(a, list) & isinstance(b, pd.DataFrame))):
        print("Error: there are lists and dataframes mixed")
        
    if ((isinstance(a, pd.DataFrame) == False) & (isinstance(a, list) == False) or
        (isinstance(b, pd.DataFrame) == False) & (isinstance(b, list) == False)):
        print("Error: input data is not a dataframe or list")
        return
    #Check que input sea válido:
    if ((isinstance(a, pd.DataFrame) == False) & (isinstance(a, list) == False) or
        (isinstance(b, pd.DataFrame) == False) & (isinstance(b, list) == False)):
        print("Error: input data is not a dataframe or list")
        return
    else:
        #Separa por tipo de input
        if dataframe == True:
            #Revisa si tienen las mismas dimensiones
            if a.shape == b.shape:
                print("Idénticas dimensiones:", a.shape)
            else:
                print("Tiene distintas dimensiones:", a.shape, "", b.shape)

            #Chequea si tienen igual valor global
            if a.mean().sum() !=  b.mean().sum():
                print("Tienen valor global distinto")
            else:
                print("Idéntico valor global")
            #Check columnas
            if list(a.columns) == list(b.columns): #tienen identicas columnas?
                #Check que las columnas estén en el mismo orden
                k = 0
                not_in_order = []
                for i in range(len(cols)):
                   if cols[i] != cols_v[i]:
                    k +=1
                    not_in_order.append(cols[i])
                if k > 0:
                    print("Hay {} columnas que no están en orden".format(k))
                    print("Columnas desordenadas:", not_in_order)

            else:
                print("El primer data tiene las siguientes columnas diferentes con el segundo:", 
                      list(set(a.columns) - set(b.columns)))
                print("")
                print("El segundo data tiene las siguientes columnas diferentes con el primero:", 
                      list(set(b.columns) - set(a.columns)))


        else:

            #Revisa si tienen las mismas dimensiones
            print("Son listas")
            if len(list(a)) == len(list(b)):
                print("Idénticas dimensiones:", len(list(a)))
            else:
                print("Tiene distintas dimensiones:", len(list(a)) , "", len(list(b)))

            #Check columnas
            if list(a) == list(b):
                print("Las listas son idénticas en todo")
              
            else:
                print("El primer data tiene las siguientes columnas diferentes con el segundo:", list(set(a) - set(b)))
                print("El segundo data tiene las siguientes columnas diferentes con el primero:", list(set(b) - set(a)))
                 #Check que las columnas estén en el mismo orden
                k = 0
                not_in_order = []
                for i in range(len(a)):
                   if a[i] != b[i]:
                    k +=1
                    not_in_order.append(a[i])
                if k > 0:
                    print("Hay {} columnas que no están en orden".format(k))
                    print("Columnas desordenadas:", not_in_order)
            

            


# In[ ]:




