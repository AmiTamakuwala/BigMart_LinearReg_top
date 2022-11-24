import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from load import (combine_name, concat, find_uniques, frequency_each_item,
                  name_dict, test, train)

import sklearn           
from sklearn import metrics

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import Lasso
from sklearn.linear_model import HuberRegressor
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import TheilSenRegressor
# from sklearn.linear_model import QuantileRegressor
from sklearn.model_selection import GridSearchCV


from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_absolute_error,mean_squared_error, r2_score

# test, train, find_uniques


# test = pd.read_csv("D:\\SalesPred\\Input\\test_data\\test_t02dQwI.csv")
# train = pd.read_csv("D:\\SalesPred\\Input\\train_data\\train_kOBLwZA.csv")

# creted function to merge both dataset:




if __name__ == "__main__":
#print(df.shape)
    # test = pd.read_csv("D:\\SalesPred\\Input\\test_data\\test_t02dQwI.csv")
    # train = pd.read_csv("D:\\SalesPred\\Input\\train_data\\train_kOBLwZA.csv")

    df = concat(train, test)
    print(df.shape)
    
    df.head()

    # Null values:
    print(df.isnull().sum())

    # find_uniques(df)
    print(find_uniques(df))
    
    # frequency of each item:
    print(frequency_each_item)

    print(df.head(10))

    print(df[df['Item_Identifier'] == 'FDP10'])

    # from "Item_Fat_Content" calling total count of different category:

    print(df["Item_Fat_Content"].value_counts())

    # calling "combine_name":
    combine_name(df, "Item_Fat_Content", name_dict)
    print(df["Item_Fat_Content"].value_counts())

    ##################   Sanity Check   ########################

    df.info()

    ### Handling Missing values:----------------------

    df.groupby(["Outlet_Type", "Outlet_Size"])["Outlet_Size"].count()
    mode_Outlet_Size = df.groupby(["Outlet_Type"])["Outlet_Size"].agg(pd.Series.mode)
    print(mode_Outlet_Size)

    print(mode_Outlet_Size.loc["Grocery Store"])

    # checking null values form the "Outlet_Size":

    bool2 = df["Outlet_Size"].isnull()
    print(df['Outlet_Size'] [bool2])

    # replacing NaN values with "Small" 
    # as "Grocery Store" has most of Small Type:

    df["Outlet_Size"][bool2] = df['Outlet_Type'][bool2].apply(lambda x: mode_Outlet_Size.loc[x]).values
    print(df['Outlet_Size'][bool2])

    # just checking total null values in "Outlet_Size" column:
    # answer will be 0 as we replace the value with "Small" 
    print(df["Outlet_Size"].isnull().sum())

    print(df["Outlet_Type"][bool2])

    ########    Handling Missing Values for "Item_Weight":

    print(df.head(10))

    # check total NaN Values in "Item_Weight" column.

    bool_values = df["Item_Weight"].isnull()
    print(df["Item_Weight"][bool_values])

   # check "Item_Idenitifier" col 
   # where NaN values are in the "Item_Weight":

    print(df["Item_Identifier"][bool_values])

    # check mean of "Item_Weight" col:

    avg_item_weight = df.groupby('Item_Identifier')['Item_Weight'].agg(np.mean)
    print(avg_item_weight)
    
    # check null values in "Item_Weight":

    df["Item_Weight"][bool_values] = df["Item_Identifier"][bool_values].apply(lambda x: avg_item_weight.loc[x]).values
    print(df['Item_Weight'].isnull().sum())

    print(df.sample(8))

    # Reduding food category to only 3 types
    #  with the help of the first 2 alphabets from the "Item_Identifier" col:

    df['Item_Type_Combine'] = df['Item_Identifier'].apply(lambda x: x[0:2])
    df['Item_Type_Combine'].replace(to_replace= ['FD', 'DR', 'NC'],
    value=["Food", "Drinks", "Non_Consumable"], inplace=True)

    # drop the "Item_Type" column as we already created "Item_Type_Combine" col:

    df = df.drop(columns=["Item_Type"])
    print(df.head())
    print(df.columns)

    # no here, all Items has 2 category in Item_Fat_Content 
    # but here, sum items are non_consumable. So , we should add one more category for non_consumable items.

    # calculating total no of "item_fat_content" and convert non_consumable item into third category Non_edible:
    
    # checking the "Non_consumable" item from "Item_type_Combine" col:
    
    bool3 = df['Item_Type_Combine'] == 'Non_Consumable'
    print(df["Item_Type_Combine"].value_counts())

    df["Item_Fat_Content"][bool3] = 'Non-edible'
    print(df['Item_Fat_Content'].value_counts())

    #########################   FEATURE ENGINEERING     ##################

    # work on "Outlet_Establishment_Year" column. 
    # (droping it and add new col "years_Old")

    df["Years_Old"] = 2013 - df['Outlet_Establishment_Year']
    df = df.drop(columns=['Outlet_Establishment_Year'])
    print(df.head())
    print(df.columns)

    print(df.info())

    # converting all the zero values to mean in the visiblity column.

    Item_Visibility_mean = df.groupby("Item_Identifier") ['Item_Visibility'].mean()
    bool4 = df['Item_Visibility'] == 0
    df['Item_Visibility'][bool4] = df['Item_Identifier'][bool4].apply(lambda x: Item_Visibility_mean.loc[x]).values
    print(df.head())
    print(df.columns)
    print(df['Item_Visibility'].head(10))

    # # Check correlation:

    # print(df.corr())

    # ###############     OUTLIERS        #################

    # # Identifying Outliers and fixing them.

    print(df.describe())

    # # creating box_plot for ouliers:

    #sns.set(style = "whitegrid")
    #sns.set(style = "whitegrid")
    sns.set(style = "whitegrid")
    ax = sns.boxplot(x=df['Item_Outlet_Sales'])

    # plotting graph for more0   Analysis:

    # plt.scatter(df.Item_MRP, df.Item_Outlet_Sales, c = 'r')
    # plt.show()

    # print(df.columns)
    # #let's go with categorical values:

    # sns.FacetGrid(df, col='Item_Type_Combine', height=10, col_wrap=5).map(plt.hist, "Item_Outlet_Sales").add_legend()
    # plt.show()

    # sns.FacetGrid(df, col='Outlet_Location_Type', height=10, col_wrap=5).map(plt.hist, "Item_Outlet_Sales").add_legend()
    # plt.show()

    # sns.FacetGrid(df, col='Outlet_Size', height=10, col_wrap=5).map(plt.hist, "Item_Outlet_Sales").add_legend()
    # plt.show()

    # sns.FacetGrid(df, col='Item_Fat_Content', height=10, col_wrap=5).map(plt.hist, "Item_Outlet_Sales").add_legend()
    # plt.show()

    # sns.FacetGrid(df, col='Outlet_Type', height=10, col_wrap=5).map(plt.hist, "Item_Outlet_Sales").add_legend()
    # plt.show()

    # Label Encoding:
    # label Encoding for all the columns with text entries
    # and we will drop the "Item_Identifier" col.

    print(df.head(3))

    le = LabelEncoder()
    list = ["Item_Fat_Content", "Outlet_Size", "Outlet_Location_Type",
            "Outlet_Type", "Item_Type_Combine"]
    
    for i in list:
        le.fit(df[i])
        df[i] = le.transform(df[i])

    df_new = df.drop(columns="Item_Identifier") # frequency encoding.
    df_new = pd.get_dummies(df_new, columns=["Outlet_Identifier"])
    print(df_new.head())
    print(df_new.columns)

    # check correlation:
    print(df.corr())

    # creating heatmap:

    sns.heatmap(df.corr())
    # plt.show()

    print(df_new.iloc[:8523, :])

    ###############     Test & Train        ######################
    # Seprating test & train dataset at the "80-20%" ratio:

    df_new_train = df_new.iloc[:8523, :]
    df_new_test = df_new.iloc[8523:, :]

    df_new_test = df_new_test.drop(columns=["Item_Outlet_Sales"])

    Y_train = df_new_train ["Item_Outlet_Sales"]
    df_train_test = df_new_train.drop(columns=['Item_Outlet_Sales'])

    print(df_train_test.shape)
    print(df_train_test.columns)

    models = [('lr', LinearRegression()),
                ('sgd', SGDRegressor()),
                ('lasso', Lasso()),
                ('ridge', Ridge()),
                ('en', ElasticNet()),
                ('huber', HuberRegressor()),
                ('ransac', RANSACRegressor()),
                ('theilSen', TheilSenRegressor())
                ]

    # sklearn.metrics.get_scorer_names()
    # sklearn.model_selection.cross_val_validate()

    def basic_model_selection(x, y, cross_folds, model):
        scores = []
        names = []
        for i, j in model:
            cv_scores = cross_val_score(j, x, y, cv=cross_folds, n_jobs=5)
            scores.append(cv_scores)
            names.append(i)
        for k in range(len(scores)):
            print(names[k], scores[k].mean())

    basic_model_selection(df_train_test, Y_train, 4, models)
    
    # MSE Score:

    basic_model_selection(df_train_test, Y_train, 4, models)

    # R2 Score:
    
    basic_model_selection(df_train_test, Y_train, 4, models)

    ####     Standardization Of the model before training        #### 

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()

    standarized = scaler.fit_transform(df_train_test)
    column_names = df_train_test.columns
    df_standardized = pd.DataFrame(data=standarized, columns=column_names)
    print(df_standardized.head())

    basic_model_selection(df_standardized, Y_train, 4, models)

    #########         ROBUST SCALER     ###############

    # * Robust Scaler handles the Outliers as well.
    # * It Scales according to the quartile range.

    from sklearn.preprocessing import RobustScaler
    from sklearn.preprocessing import MinMaxScaler

    normalize = MinMaxScaler()
    robust = RobustScaler(quantile_range=(0.1, 0.8))

    robust_stan = robust.fit_transform(df_train_test)
    robust_stan_normalize = normalize.fit_transform(robust_stan)

    # also normalized the dataset using MinMaxScaler 
    # i.e. has bought the dataset between(0,1)

    df_robust_normalize = pd.DataFrame(robust_stan_normalize, columns = column_names)
    print(df_robust_normalize.head())

    basic_model_selection(df_robust_normalize, Y_train, 4, models)

    ##########      BEST MODEL      ###############################

    robust_test = robust.fit_transform(df_new_test)
    robust_normalize_test = normalize.fit_transform(robust_test)
    df_test_robust_normalize = pd.DataFrame(robust_normalize_test, columns= column_names)

    model = LinearRegression()

    print(model.fit(df_robust_normalize, Y_train))

    ######  Metrics Calculation:    ###########

    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

    import math

    def root_mean_squared_error(y, y_pred):
        mse = np.square(np.subtract(y, y_pred)).mean()
        rmse = math.sqrt(mse)
        return rmse

    print(model.intercept_, model.coef_)

    print("MAE Score for model is: ", mean_absolute_error(Y_train, model.predict(df_robust_normalize)))
    print("MSE Score for model is: ", mean_squared_error(Y_train, model.predict(df_robust_normalize)))
    print("RMSE Score for model is: ", root_mean_squared_error(Y_train, model.predict(df_robust_normalize)))
    print("R2 Score for model is: ", r2_score(Y_train, model.predict(df_robust_normalize)))

    final_prediction = model.predict(df_test_robust_normalize)

    df_final_prediction = pd.DataFrame(final_prediction, columns=['Item_Outlet_Sales'])

    print(df_final_prediction.head())

    ##########      SAVING THE FINAL MNODEL USING JOBLIB    ########

    import joblib
    filename = 'linear_regression.sav'
    print(joblib.dump(model, filename))

    # this command loads the model once again.

    load_model = joblib.load(filename)

    float_formatter = "{:.2f}".format

    print(load_model.predict(np.array(df_robust_normalize.iloc[1, :]).reshape(1, -1)))
    print(load_model.predict(np.array(df_robust_normalize.iloc[0:5, :])))

    print(df.head())

    ###############         HYPERPARAMETER       ###################
    # The model for Hyperparameter tuning are same regression Models:

    # defining function for hyper parameter tuning 
    # and using RMSE as metrics.

    def model_parameter_tuning(x, y, model, parameters, cross_folds):
        model_grid = GridSearchCV(model,
                                    parameters,
                                    cv = cross_folds,
                                    n_jobs = 5,
                                    verbose = True)
        model_grid.fit(x,y)
        y_predicted = model_grid.predict(x)
        print(model_grid.score)
        print(model_grid.best_params_)
        print("The RMSE Score is : ", np.sqrt(np.mean((y- y_predicted)**2)))

    #model_parameter_tuning(df_standardized, Y_train, model, model_parameters, 4)

















    


    












































































