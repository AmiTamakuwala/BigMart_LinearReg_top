import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import featuretools as ft


from scipy import stats

import warnings
warnings.filterwarnings('ignore')

# from sklearn.linear_model import Ridge
# from sklearn.model_selection import KFold, cross_val_score
# from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import GridSearchCV
# from sklearn.model_selection import RandomizedSearchCV

test = pd.read_csv("D:\\SalesPred\\Input\\test_data\\test_t02dQwI.csv")
train = pd.read_csv("D:\\SalesPred\\Input\\train_data\\train_kOBLwZA.csv")

print(test.shape, train.shape)

def concat(X,Y):
    """
    X: test datset.
    Y: train datset.
    return: None
    """
    df=pd.concat([X,Y], ignore_index=True)
    return df
df = concat(train, test)
# print(df.shape)

def find_uniques(df):
    return df.apply(lambda x: len(x.unique()))
find_uniques(df)
# print(find_uniques(df))

#df.isnull().sum()
#print(df.isnull().sum())
    
# find_uniques(df)
# print(find_uniques(df))

def frequency_each_item(df,col):
    """
    This function prints uniques value of columns.
    df: dataframe
    col: column name.
    return: None
    """
    for i in col:
        print("Frequency of each category for: ", i)
        print(df[i].value_counts())
        print("-"*100)
col_name = ["Item_Fat_Content", "Item_Type", "Outlet_Size", "Outlet_Location_Type", "Outlet_Type"]
frequency_each_item(df, col_name)
# print(frequency_each_item)

# function for replacing "low fat", "LF", "Reg" into only 2 category 
# which will be "Low Fat" & "RFegular".

name_dict = {"low fat" : 'Low Fat',
                "LF" : "Low Fat", 
                "reg" : "Regular"}

def combine_name(df, col, values):
    return df[col].replace(values, inplace= True)

combine_name(df, "Item_Fat_Content", name_dict)
# print(df["Item_Fat_Content"].value_counts())

# name_dict = {"reg" : "Regular",
#                 "LF" : "Low Fat",
#                 "low fat" : "Low Fat"}
# def merge_str(df, col, value):
#      return df[col].replace(value, inplace=True)

# merge_str(df, "Item_Fat_Content", name_dict)    
# print(df["Item_Fat_Content"].value_counts())


















    