#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 13:40:44 2021

@author: carolyndavis
"""

import acquire as a 
import pandas as pd 
import os
import numpy as np

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
import sklearn.preprocessing

from scipy import stats

from sklearn.metrics import mean_squared_error

from sklearn.metrics import explained_variance_score

from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
from sklearn.preprocessing import PolynomialFeatures




#Imports the previously acquired data from the acquire.py file 
zillow_df = a.get_zillow_data()




# =============================================================================
#                     DATA SUMMARY
# =============================================================================
#Rename the Columns:
zillow_df.columns

#Look at the data for values 
zillow_df.info()
#fips is a unique county idenfier code

zillow_df.shape


zillow_df.describe().T




zillow_df.isnull().sum()
# =============================================================================
# TAKEAWAYS:
# =============================================================================
#rename the columns for readibility 
#make all the values ints?except transaction date

        #NULLS:
# parcelid                         0
# bedroomcnt                       0
# bathroomcnt                      0
# calculatedfinishedsquarefeet    47 <- 
# taxvaluedollarcnt                1 <-
# taxamount                        1 <-
# assessmentyear                   0
# regionidcounty                   0
# regionidzip                     17 <-
# transactiondate                  0
# dtype: int64
# (these nulls need to be replaced with Naan values )




# =============================================================================
#                         TIDY THE DATA
# =============================================================================
zillow_df.columns 
zillow_df = zillow_df.rename(columns = {'parcelid': 'parcel_id', 'bedroomcnt': 'bed_count',
                                      'bathroomcnt': 'bath_count',
                                      'calculatedfinishedsquarefeet': 'sq_feet',
                                      'taxvaluedollarcnt': 'tax_value',
                                      'yearbuilt': 'year_built', 'taxamount': 'tax_amount',
                                      'assessmentyear': 'assessment_year',
                                      'regionidcounty': 'region_county_id', 'regionidzip': 'region_zip',
                                      'transactiondate': 'transaction_date'})



#Replace a whitespace sequence or empty with a NaN value and reassign this manipulation to df:
zillow_df = zillow_df.replace(r'^\s*S', np.nan, regex=True)


#check to see if we actually have true nulls in the dataframe 
zillow_df.info()

#DROP DUPES:
zillow_df.drop_duplicates()


#DROP the Null Values:
    
zillow_df = zillow_df.dropna()
zillow_df.info()
zillow_df.describe()

################## #SPEARMAN CORRELATION COEFFICIENT VIZ: ###################################

# # Statistic measure of the strength of a montonic relationship between the paired data
# num_cols = [col for col in zillow_df.columns if (zillow_df[col].dtype in ['float64','int64'] and col not in ['parcel_id','transaction_date'])
#             or zillow_df[col].dtype.name=='category']
# temp_df = zillow_df[num_cols]
# corrmat = temp_df.corr(method='spearman')
# f, ax = plt.subplots(figsize=(12, 12))

# sns.heatmap(corrmat, vmax=1., square=True,cmap='PiYG')
# plt.title("Variables correlation map", fontsize=15)
# plt.show()


############################# HISTOGRAPH VIZ ################################################


# plt.figure(figsize=(16, 3))

# List of columns
cols = [col for col in zillow_df.columns if col not in ['parcel_id', 'transaction_date']]

for i, col in enumerate(cols):

    # i starts at 0, but plot nos should start at 1
    plot_number = i + 1 

    # Create subplot.
    plt.subplot(1, len(cols), plot_number)

    # Title with column name.
    plt.title(col)

    # Display histogram for column.
    # zillow_df[cols].hist(bins=5,ec='black')

  

    # Hide gridlines.
    plt.grid(False)
    zillow_df[cols].hist(alpha=0.5, figsize=(20, 10))
    # # turn off scientific notation
    plt.ticklabel_format(useOffset=False)
    # plt.xticks(rotation=90)
    # plt.ticklabel_format(style='plain', axis='x')
    
    plt.show()




############################ BOXPLOT VIZ ##########################################

plt.figure(figsize=(8,4))

plt.ticklabel_format(useOffset=False, style='plain')
sns.boxplot(data=zillow_df.drop(columns=['assessment_year', 'transaction_date', 'parcel_id']))

plt.show()

#Takeaways from Spearman, Histographs, and Boxplot:
    #high correlation between tax_value and tax_amount *duh*
    #moderate correlation between tax_value and bath_count/area.. (worth looking into)
    #assessment_year is not unique, consider dropping
    #Extreme outliers exist in the dataset and need to be removed
    #Can use IQR rule to remove the outliers
    

#Dropped these columns because they are not in the key requirments for the prokect.
#transaction_date and assessment_year are already pre-filtered to the specifications of "hot months"
zillow_df.drop(columns=['assessment_year', 'region_county_id', 'region_zip', 'transaction_date'], axis = 1, inplace=True)


zillow_df.columns


###################################################################################
            ##Removing the Outliers Mentioned Above:

                


#There appeared to be some extreme outliers in the data indicated in the boxplot distribution 
def remove_outliers(df, k, col_list):
    ''' remove outliers from a list of columns in a dataframe 
        and return that dataframe
    '''
    
    for col in col_list:

        q1, q3 = df[col].quantile([.01, .99])  # drops the 
        
        iqr = q3 - q1   # calculate interquartile range
        
        upper_bound = q3 + k * iqr   # get upper bound
        lower_bound = q1 - k * iqr   # get lower bound

        # return dataframe without outliers
        
        df = df[(df[col] > lower_bound) & (df[col] < upper_bound)]
        
    return df




zillow_df = remove_outliers(zillow_df,1.5, ['bed_count', 'bath_count', 'sq_feet', 'tax_value',
       'tax_amount'])
zillow_df.columns



zillow_df.info()





######################## Check to See Now that Outliers Are Removed ######################

plt.figure(figsize=(16, 3))

# List of columns
cols = [col for col in zillow_df.columns if col not in ['assessment_year', 'transaction_date', 'parcel_id', 'region_county_id']]

for i, col in enumerate(cols):

    # i starts at 0, but plot nos should start at 1
    plot_number = i + 1 

    # Create subplot.
    plt.subplot(1, len(cols), plot_number)

    # Title with column name.
    plt.title(col)

    # Display histogram for column.
    zillow_df[col].hist(bins=5)

    # Hide gridlines.
    plt.grid(False)
    
    # turn off scientific notation
    plt.ticklabel_format(useOffset=False)
    
    # mitigate overlap
    plt.tight_layout()
    
plt.show()

####### TAKEAWAYS:
    #bed_count, bath_count, and area are not normally distributed
    
# zillow_df.info()
# zillow_df['region_county_id'].value_counts() #possible new feature

# =============================================================================
# Visualizing the Three County Codes 
# =============================================================================


##################    SCATTERPLOT VIZ OF DISTRIBUTION, Tax Amount and Region_county_zip
sns.scatterplot(x=zillow_df['sq_feet'], y=zillow_df['tax_value'], data=zillow_df)

###################     HISTOGRAPH VIZ
zillow_df['sq_feet'].hist(bins=5, ec='black')

zillow_df['bed_count'].hist(bins=5, ec='black')

zillow_df['bath_count'].hist(bins=5, ec='black')
##################   BOXPLOT VIZ ###########
# plt.figure(figsize=(8,4))

# plt.ticklabel_format(useOffset=False, style='plain')
# sns.boxplot(data=zillow_df[reg])

# plt.show()


# num_cols = [i for i in zillow_df.region_county_id if zillow_df[i].dtype.name=='category']
# temp_df = zillow_df[num_cols]
# corrmat = temp_df.corr(method='spearman')
# f, ax = plt.subplots(figsize=(12, 12))

# sns.heatmap(corrmat, vmax=1., square=True,cmap='PiYG')
# plt.title("County Code correlation map", fontsize=15)
# plt.show()





######## TAKEAWAYS:
    #Scatterplot suggests
    #There is likely a correlation between the target and region_county_id
    #There are some extreme outliers in the region zip for county code 3101
    #It is worth exploring any connection between the three variables correlated to 
    #the target (area, num_baths, region_county_id)
    #say something about scaling 




# =============================================================================
#                     SPLITTING INTO TRAIN VALIDATE TEST 
# =============================================================================
# get value counts and decide on data types
cols = zillow_df.columns

for col in cols:
    
    print(col.upper())
    print(zillow_df[col].value_counts())
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print()
    
    
zillow_df.info()


#Tax value dollar count is the target
#tax amount needs to be dropped as well prior 



zillow_df.columns



#Need to remove the other columns to focus on the MVP:
    
split_df = zillow_df[['bed_count', 'bath_count', 'sq_feet', 'tax_value']]



# =============================================================================
# The function below will now run on just our focus cols, dropping the target where needed
# =============================================================================

#Need to remove the other columns to focus on the MVP:
    
split_df = zillow_df[['bed_count', 'bath_count', 'sq_feet', 'tax_value']]


def train_validate_test(df, target):
    '''
    this function takes in a dataframe and splits it into 3 samples,
    a test, which is 20% of the entire dataframe,
    a validate, which is 24% of the entire dataframe,
    and a train, which is 56% of the entire dataframe.
    It then splits each of the 3 samples into a dataframe with independent variables
    and a series with the dependent, or target variable.
    The function returns train, validate, test sets and also another 3 dataframes and 3 series:
    X_train (df) & y_train (series), X_validate & y_validate, X_test & y_test.
    '''
    # split df into test (20%) and train_validate (80%)
    train_validate, test = (train_test_split(df, test_size=.2, random_state=123))
   
    # split train_validate off into train (70% of 80% = 56%) and validate (30% of 80% = 24%)
    train, validate = train_test_split(train_validate, test_size=.3, random_state=123)
    
    # split train into X (dataframe, drop target) & y (series, keep target only)
    X_train = train.drop(columns=[target])
    y_train = train[target]
    
    # split validate into X (dataframe, drop target) & y (series, keep target only)
    X_validate = validate.drop(columns=[target])
    y_validate = validate[target]
    
    # split test into X (dataframe, drop target) & y (series, keep target only)
    X_test = test.drop(columns=[target])
    y_test = test[target]
    
    return train, validate, test, X_train, y_train, X_validate, y_validate, X_test, y_test



train, validate, test, X_train, y_train, X_validate, y_validate, X_test, y_test = train_validate_test(split_df, target='tax_value')



#Checking the size of the samples 
train.shape
# (15699, 4)

test.shape
# (5608, 4)

validate.shape
# (6729, 4)


# =============================================================================
#                     SCALING THE DATA 
# =============================================================================
# # Utilized a MinMaxScaller for the data to transform each value in the column 
# proprtionately with the desirable range 0 and 1. 
#Additionally we are dealing with different units for values (dollar, sq ft)


def Min_Max_Scaler(X_train, X_validate, X_test):
    """
    Takes in X_train, X_validate and X_test dfs with numeric values only
    Returns scaler, X_train_scaled, X_validate_scaled, X_test_scaled dfs 
    """
    #Fit the thing
    scaler = sklearn.preprocessing.MinMaxScaler().fit(X_train)
    
    #transform the thing
    X_train_scaled = pd.DataFrame(scaler.transform(X_train), index = X_train.index, columns = X_train.columns)
    X_validate_scaled = pd.DataFrame(scaler.transform(X_validate), index = X_validate.index, columns = X_validate.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), index = X_test.index, columns = X_test.columns)
    
    return scaler, X_train_scaled, X_validate_scaled, X_test_scaled


scaler, X_train_scaled, X_validate_scaled, X_test_scaled = Min_Max_Scaler(X_train, X_validate, X_test)

#--------VISUALIZE THE SCALED DATA
def visualize_scaled_date(scaler, scaler_name, feature):
    scaled = scaler.fit_transform(train[[feature]])
    fig = plt.figure(figsize = (12,6))

    gs = plt.GridSpec(2,2)

    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1,0])
    ax3 = fig.add_subplot(gs[1,1])

    ax1.scatter(train[[feature]], scaled)
    ax1.set(xlabel = feature, ylabel = 'Scaled' + feature, title = scaler_name)

    ax2.hist(train[[feature]])
    ax2.set(title = 'Original')

    ax3.hist(scaled)
    ax3.set(title = 'Scaled')
    plt.tight_layout();



# Visualize scaling data for bed_count

visualize_scaled_date(sklearn.preprocessing.MinMaxScaler(), 'Min Max Scaler', 'bed_count')


# Visualize scaling data for bath_count


visualize_scaled_date(sklearn.preprocessing.MinMaxScaler(), 'Min Max Scaler', 'bath_count')



# Visualize scaling data for area


visualize_scaled_date(sklearn.preprocessing.MinMaxScaler(), 'Min Max Scaler', 'sq_feet')




# =============================================================================
#             SCALING FOR POSSIBLE OUTLIERS : MINMAX SCALER
# =============================================================================
#Train dataframe, but with outliers removed
train_no_outliers = train[train.tax_value <= 2_000_000]


#scale values of 'tax_values' with outliers
scaler2 = sklearn.preprocessing.MinMaxScaler()
scaled2 = scaler.fit_transform(train[['tax_value']])
scaled2



fig = plt.figure(figsize = (18,5))

plt.subplot(131)
plt.hist(train.tax_value, bins = 30)
plt.title('Unscaled')
# plt.xlim(-1,20)

plt.subplot(132)
plt.hist(scaled2, bins = 30)
plt.title('Min-Max with outliers')
# plt.xlim(-1,20)

plt.subplot(133)
plt.hist(scaled2)
plt.title('Min-Max without outliers')

#--------------------------------------------------------
fig = plt.figure(figsize = (12,6))
plt.subplot(121)
plt.scatter(train.sq_feet, train.tax_value, c = train.tax_value,cmap = 'plasma_r')
plt.gcf().axes[0].yaxis.get_major_formatter().set_scientific(False)
plt.title('Unscaled')
plt.xlabel('Square feet/Area')
plt.ylabel('tax value')


plt.subplot(122)
plt.scatter(train.sq_feet, scaled2, c = train.tax_value,cmap = 'plasma_r')
plt.gcf().axes[0].yaxis.get_major_formatter().set_scientific(False)
plt.title('Min Max scaled')
plt.xlabel('Square feet/Area')
plt.ylabel('Scaled tax value')





# =============================================================================
#                         EXPLORE 
# =============================================================================
corr_df = pd.concat([X_train_scaled, y_train], axis=1)

#Create a Correlation Matrix for all the features
corrs = corr_df.corr()
corrs


#             bed_count  bath_count   sq_feet  tax_value
# bed_count    1.000000    0.641932  0.625294   0.283234
# bath_count   0.641932    1.000000  0.847396   0.549609
# sq_feet      0.625294    0.847396  1.000000   0.629578
# tax_value    0.283234    0.549609  0.629578   1.000000

# plt.figure(figsize=(8,6))
# sns.heatmap(corr_df, cmap='Purples', annot=True, linewidth=0.5, mask= np.triu(corr_df))
# plt.ylim(0, 4)

# plt.show()

corr_spearman = corr_df.corr(method='spearman')
corr_spearman


plt.figure(figsize=(8,6))
sns.heatmap(corr_spearman, cmap='Purples', annot=True, linewidth=0.5, mask= np.triu(corrs))
plt.ylim(0, 4)

plt.show()

# =============================================================================
# 
# =============================================================================
sns.pairplot(corr_df[['bed_count', 'bath_count', 'sq_feet', 'tax_value']], corner=True)
plt.show()

corrs.dtypes
# =============================================================================
#                             PearsonR Stats Tests on Features
# =============================================================================
#Utilize the SciPy Stats: PearsonR  to get the correlation coefficient for area
r, p = stats.pearsonr(X_train_scaled['sq_feet'], y_train)
print(f'The p-value is: {p}. There is {round(p,5)}% chance that we see these results by chance.')
print(f'r = {round(r, 2)}')

# The p-value is: 0.0. There is 0.0% chance that we see these results by chance.
# r = 0.63

#Utilize the SciPy Stats: PearsonR  to get the correlation coefficient for bed_count
r, p = stats.pearsonr(X_train_scaled['bed_count'], y_train)
print(f'The p-value is: {p}. There is {round(p,2)}% chance that we see these results by chance.')
print(f'r = {round(r, 2)}')

# # The p-value is: 2.0903268105279885e-287. There is 0.2% chance that we see these results by chance.
# r = 0.28

#Utilize the SciPy Stats: PearsonR  to get the correlation coefficient for bath_count
r, p = stats.pearsonr(X_train_scaled['bath_count'], y_train)
print(f'The p-value is: {p}. There is {round(p,5)}% chance that we see these results by chance.')
print(f'r = {round(r, 2)}')

# The p-value is: 0.0. There is 0.0% chance that we see these results by chance.
# r = 0.55

# TakeAWAYS:
#     HYP1, HYP2, HYP3: Reject null hypothesis
#the p values are low so drop the nulls. All three features are worth exploring





#Add in some heatmap visuals/spearman?





# =============================================================================
#                             MODELING
# =============================================================================
# =============================================================================
# ESTABLISH THE BASELINE:
# =============================================================================




plt.hist(y_train)
plt.xlabel("Tax Value")
plt.ylabel("Amount")
plt.show()

round((y_train.mean()), 2)
#521221.76
round(y_train.median(), 2)
# 383995.0




# We need y_train and y_validate to be dataframes to append the new columns with predicted values. 
y_train_df = pd.DataFrame(y_train)
y_validate_df = pd.DataFrame(y_validate)

# 1. Predict z_pred_mean
z_pred_mean = y_train_df['tax_value'].mean()
y_train_df['z_pred_mean'] = z_pred_mean
y_validate_df['z_pred_mean'] = z_pred_mean

# 2. compute z_pred_median
z_pred_median = y_train_df['tax_value'].median()
y_train_df['z_pred_median'] = z_pred_median
y_validate_df['z_pred_median'] = z_pred_median

# 3. RMSE of z_pred_mean
rmse_train = mean_squared_error(y_train_df.tax_value, y_train_df.z_pred_mean)**(.5)
rmse_validate = mean_squared_error(y_validate_df.tax_value, y_validate_df.z_pred_mean)**(.5)

print("RMSE using Mean\nTrain/In-Sample: ", round(rmse_train, 2), 
      "\nValidate/Out-of-Sample: ", round(rmse_validate, 2))

# 4. RMSE of z_pred_median
rmse_train = mean_squared_error(y_train_df.tax_value, y_train_df.z_pred_median)**(.5)
rmse_validate = mean_squared_error(y_validate_df.tax_value, y_validate_df.z_pred_median)**(.5)

print("RMSE using Median\nTrain/In-Sample: ", round(rmse_train, 2), 
      "\nValidate/Out-of-Sample: ", round(rmse_validate, 2))

# RMSE using Mean
# Train/In-Sample:  576143.18 
# Validate/Out-of-Sample:  554977.96
# RMSE using Median
# Train/In-Sample:  592260.2 
# Validate/Out-of-Sample:  567774.49

#Looking good, model perfromed way better than the baseline for mean

#Making the Metric DF

def make_metric_df(y, y_pred, model_name, metric_df):
    if metric_df.size ==0:
        metric_df = pd.DataFrame(data=[
            {
                'model': model_name, 
                'RMSE_validate': mean_squared_error(
                    y,
                    y_pred) ** .5,
                'r^2_validate': explained_variance_score(
                    y,
                    y_pred)
            }])
        return metric_df
    else:
        return metric_df.append(
            {
                'model': model_name, 
                'RMSE_validate': mean_squared_error(
                    y,
                    y_pred) ** .5,
                'r^2_validate': explained_variance_score(
                    y,
                    y_pred)
            }, ignore_index=True)


# create the metric_df as a blank dataframe
metric_df = pd.DataFrame()
# make our first entry into the metric_df with mean baseline
metric_df = make_metric_df(y_train_df.tax_value,
                           y_train_df.z_pred_mean,
                           'mean_baseline',
                          metric_df)
metric_df

#            model  RMSE_validate  r^2_validate
# 0  mean_baseline  576143.179975           0.0

#-----------------------------

# plot to visualize actual vs predicted. 
plt.hist(y_train_df.tax_value, color='blue', alpha=.5, label="Actual Property Value")
plt.hist(y_train_df.z_pred_mean, bins=1, color='red', alpha=.5, rwidth=100, label="Predicted Properyt Value - Mean")
plt.hist(y_train_df.z_pred_median, bins=1, color='orange', alpha=.5, rwidth=100, label="Predicted Property Value  - Median")
plt.xlabel("Property Value")
plt.ylabel("Amount")
plt.legend()
plt.show()


# =============================================================================
# MODELING
# =============================================================================
                        #Linear Regression (OLS)
                        
# create the model object
lm = LinearRegression(normalize=True)

# fit the model to our training data. We must specify the column in y_train, 
# since we have converted it to a dataframe from a series! 
lm.fit(X_train_scaled, y_train_df.tax_value)

# predict train
y_train_df['z_pred_lm'] = lm.predict(X_train_scaled)

# evaluate: rmse
rmse_train = mean_squared_error(y_train_df.tax_value, y_train_df.z_pred_lm)**(1/2)

# predict validate
y_validate_df['z_pred_lm'] = lm.predict(X_validate_scaled)

# evaluate: rmse
rmse_validate = mean_squared_error(y_validate_df.tax_value, y_validate_df.z_pred_lm)**(1/2)

print("RMSE for OLS using LinearRegression\nTraining/In-Sample: ", rmse_train, 
      "\nValidation/Out-of-Sample: ", rmse_validate)


# RMSE for OLS using LinearRegression
# Training/In-Sample:  438219.89832767576 
# Validation/Out-of-Sample:  426324.5228746647

metric_df = metric_df.append({
    'model': 'OLS Regressor', 
    'RMSE_validate': rmse_validate,
    'r^2_validate': explained_variance_score(y_validate_df.tax_value, y_validate_df.z_pred_lm)}, ignore_index=True)

metric_df

#            model  RMSE_validate  r^2_validate
# 0  mean_baseline  576143.179975      0.000000
# 1  OLS Regressor  426324.522875      0.409793

#------------------------------------------
                        #LassoLars
# create the model object
lars = LassoLars(alpha=1.0)

# fit the model to our training data. We must specify the column in y_train, 
# since we have converted it to a dataframe from a series! 
lars.fit(X_train_scaled, y_train_df.tax_value)

# predict train
y_train_df['z_pred_lars'] = lars.predict(X_train_scaled)

# evaluate: rmse
rmse_train = mean_squared_error(y_train_df.tax_value, y_train_df.z_pred_lars)**(.5)

# predict validate
y_validate_df['z_pred_lars'] = lars.predict(X_validate_scaled)

# evaluate: rmse
rmse_validate = mean_squared_error(y_validate_df.tax_value, y_validate_df.z_pred_lars)**(.5)

print("RMSE for Lasso + Lars\nTraining/In-Sample: ", rmse_train, 
      "\nValidation/Out-of-Sample: ", rmse_validate)


# RMSE for Lasso + Lars
# Training/In-Sample:  438220.0079189707 
# Validation/Out-of-Sample:  426316.0785837816



metric_df = make_metric_df(y_validate_df.tax_value,
               y_validate_df.z_pred_lars,
               'lasso_alpha_1',
               metric_df)
metric_df




#            model  RMSE_validate  r^2_validate
# 0  mean_baseline  576143.179975      0.000000
# 1  OLS Regressor  426324.522875      0.409793
# 2  lasso_alpha_1  426316.078584      0.409817
# =============================================================================
#                                 EVALUATE 
# =============================================================================

# PLOTTING  ACTUAL VS PREDICTED VALUES


# y_validate.head()
plt.figure(figsize=(16,8))
plt.plot(y_validate_df.tax_value, y_validate_df.z_pred_mean, alpha=.5, color="gray", label='_nolegend_')
plt.annotate("Baseline: Predict Using Mean", (16, 9.5))
plt.plot(y_validate_df.tax_value, y_validate_df.tax_value, alpha=.5, color="blue", label='_nolegend_')
plt.annotate("The Ideal Line: Predicted = Actual", (.5, 3.5), rotation=15.5)

plt.scatter(y_validate_df.tax_value, y_validate_df.z_pred_lm, 
            alpha=.5, color="red", s=100, label="Model: Linear Regression")
plt.scatter(y_validate_df.tax_value, y_validate_df.z_pred_lars, 
            alpha=.5, color="yellow", s=100, label="Model: LASSO + LARS")
plt.legend()
plt.xlabel("Actual Property Tax Value")
plt.ylabel("Predicted Property Tax Value")
plt.title("Model Evaluation")
# plt.annotate("The polynomial model appears to overreact to noise", (2.0, -10))
# plt.annotate("The OLS model (LinearRegression)\n appears to be most consistent", (15.5, 3))
plt.show()



# Residual Plots: Plotting the Errors in Predictions

# y_validate.head()
plt.figure(figsize=(16,8))
plt.axhline(label="No Error")
plt.scatter(y_validate_df.tax_value, y_validate_df.z_pred_lm - y_validate_df.tax_value , 
            alpha=.5, color="red", s=100, label="Model: Linear Regression")
plt.scatter(y_validate_df.tax_value, y_validate_df.z_pred_lars - y_validate_df.tax_value, 
            alpha=.5, color="yellow", s=100, label="Model: LASSO + LARS")

plt.legend()
plt.xlabel("Actual Property Tax Value")
plt.ylabel("Residual/Error: Predicted Property Tax Value - Actual ")
plt.title("Plotting the Errors in Predictions")
# plt.annotate("The polynomial model appears to overreact to noise", (2.0, -10))
# plt.annotate("The OLS model (LinearRegression)\n appears to be most consistent", (15.5, 3))
plt.show()

# Histograms 

# plot to visualize actual vs predicted. 
plt.figure(figsize=(16,8))
plt.hist(y_validate_df.tax_value, color='blue', alpha=.5, label="Actual Property Tax Value")
plt.hist(y_validate_df.z_pred_lm, color='red', alpha=.5, label="Model: OLS")
plt.hist(y_validate_df.z_pred_lars, color='green', alpha=.5, label="Model: LASSO + LARS")
plt.xlabel("Actual Property Tax Value ")
plt.ylabel("Predicted Property Tax Value")
plt.title("Comparing the Distribution of Actual Tax Values to Predicted for the Top Models")
plt.legend()
plt.show()

########## COMPARING THE MODELS DF:
metric_df[['model', 'RMSE_validate']]

#            model  RMSE_validate
# 0  mean_baseline  576143.179975
# 1  OLS Regressor  426324.522875
# 2  lasso_alpha_1  426316.078584




# =============================================================================
#                     MODEL SELECTION
# =============================================================================
#-----------------------------------------------------------------------------
# Model Selected: lm using Linear Regression
#-----------------------------------------------------------------------------
y_test = pd.DataFrame(y_test)

# predict on test
y_test['z_pred_lm'] = lm.predict(X_test_scaled)

# evaluate: rmse
rmse_test = mean_squared_error(y_test.tax_value, y_test.z_pred_lm) ** (.5)

print("RMSE for OLS Model using LinearRegression\nOut-of-Sample Performance: ", rmse_test)

# =============================================================================
# RMSE for OLS Model using LinearRegression
# Out-of-Sample Performance:  436703.5099971608
# =============================================================================




#-----------------------------------------------------------------------------
#Model SELECTED LASSO LARS
#-----------------------------------------------------------------------------
y_test = pd.DataFrame(y_test)

# predict on test
y_test['z_pred_lars'] = lars.predict(X_test_scaled)

# evaluate: rmse
rmse_test = mean_squared_error(y_test.tax_value, y_test.z_pred_lars) ** (.5)

print("RMSE for LARS Model using LARSLASSO\nOut-of-Sample Performance: ", rmse_test)

# =============================================================================
# RMSE for LARS Model using LARSLASSO
# Out-of-Sample Performance:  436702.0948379894
# =============================================================================

#Takeaways:
    
    #These models both perform extremely similar on the test 
    #Both models perform better than the baseline 
    #OLS model perform 1$ better than LARS LASSO on prediction of property values
    






#-----------------------------------------------------------------------------

# zillow_df['region_county_id'].value_counts()



# counties = {3101: 'LA', 1286: 'Orange', 2061: 'Ventura'}

# zillow_df['county'] = zillow_df.region_county_id.replace(counties)
# zillow_df['tax_rate'] = round((zillow_df.tax_amount / zillow_df.tax_value) *100,2)

# plt.figure(figsize=(14,10))
# sns.histplot(data=(zillow_df), x='tax_rate', hue='county', bins = 1000)
# plt.xlim(0.75, 3)
# plt.xticks(fontsize = 15)
# plt.yticks(fontsize = 15)
# plt.xlabel('Tax Rate Percentage', fontsize = 15)
# plt.ylabel('Frequency', fontsize = 15)
# plt.title('Distributions of Tax Rates by County', x = .24, fontsize = 20)



# zillow_df.groupby('county').tax_rate.mean().round(2)

##############################################################################
                        #SCRAP CODE

# def correlation_exploration(df, x_string, y_string):
#     '''
#     This function takes in a df, a string for an x-axis variable in the df, 
#     and a string for a y-axis variable in the df and displays a scatter plot, the r-
#     squared value, and the p-value. It explores the correlation between input the x 
#     and y variables.
#     '''
#     r, p = stats.pearsonr(X_train_scaled[x_string], y_train_df[y_string])
#     df.plot.scatter(x_string, y_string)
#     plt.title(f"{x_string}'s Relationship with {y_string}")
#     print(f'The p-value is: {p}. There is {round(p,3)}% chance that we see these results by chance.')
#     print(f'r = {round(r, 2)}')
#     plt.show()
#     plt.close()

# for col in corr_df.columns[0:3]:
#     correlation_exploration(corr_df, col, 'tax_value')


