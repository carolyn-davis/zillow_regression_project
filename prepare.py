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

import seaborn as sns
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
                                      'calculatedfinishedsquarefeet': 'area',
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

# Statistic measure of the strength of a montonic relationship between the paired data
num_cols = [col for col in zillow_df.columns if (zillow_df[col].dtype in ['float64','int64'] and col not in ['parcel_id','transaction_date'])
            or zillow_df[col].dtype.name=='category']
temp_df = zillow_df[num_cols]
corrmat = temp_df.corr(method='spearman')
f, ax = plt.subplots(figsize=(12, 12))

sns.heatmap(corrmat, vmax=1., square=True,cmap='PiYG')
plt.title("Variables correlation map", fontsize=15)
plt.show()


############################# HISTOGRAPH VIZ ################################################


plt.figure(figsize=(16, 3))

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
    zillow_df[col].hist(bins=5,ec='black')

    # Hide gridlines.
    plt.grid(False)
    
    # turn off scientific notation
    plt.ticklabel_format(useOffset=False)
    
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

###################################################################################
            ##Removing the Outliers Mentioned Above:

def remove_outliers(df, k, col_list):
    ''' remove outliers from a list of columns in a dataframe 
        and return that dataframe
    '''
    
    for col in col_list:

        q1, q3 = df[col].quantile([.25, .75])  # get quartiles
        
        iqr = q3 - q1   # calculate interquartile range
        
        upper_bound = q3 + k * iqr   # get upper bound
        lower_bound = q1 - k * iqr   # get lower bound

        # return dataframe without outliers
        
        df = df[(df[col] > lower_bound) & (df[col] < upper_bound)]
        
    return df

zillow_df = remove_outliers(zillow_df,1.5, ['bed_count', 'bath_count', 'area', 'tax_value',
       'tax_amount', 'region_county_id', 'region_zip'])
zillow_df.columns



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