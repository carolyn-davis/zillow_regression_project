#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 11:52:08 2021

@author: carolyndavis
"""

# =============================================================================
# ACQUIRE.PY SCRIPT FOR ZILLOW DATA: REGRESSION PROJECT:
# =============================================================================
    
# IMPORTS UTILIZED FOR IMPORT OF ZILLOW DATA
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from env import host, user, password
import os
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer





def get_db_url(db, user=user, host=host, password=password):
    '''
    This function uses my info from my env file to
    create a connection url to access the Codeup db.
    It takes in a string name of a database as an argument.
    '''
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'

# =============================================================================
# 
# =============================================================================

def new_zillow_data():
    '''
    This function reads the zillow data from the Codeup db into a df,
    write it to a csv file, and returns the df.
    '''
    # Create SQL query.
    sql_query =  """
            
    SELECT parcelid, bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet, taxvaluedollarcnt, taxamount, assessmentyear, regionidcounty,regionidzip, transactiondate
    FROM properties_2017
    LEFT JOIN propertylandusetype USING(propertylandusetypeid)
    JOIN predictions_2017 USING(parcelid)
    WHERE propertylandusedesc IN ("Single Family Residential",
                              "Inferred Single Family Residential")
                              AND (transactiondate BETWEEN '2017-05-01' AND '2017-08-31')"""
    
    # Read in DataFrame from Codeup db.
    df = pd.read_sql(sql_query, get_db_url('zillow'))
    
    return df
#
# =============================================================================
# 
# =============================================================================

def get_zillow_data():
    '''
    This function reads i zillow data from Codeup database, writes data to
    a csv file if a local file does not exist, and returns a df.
    '''
    if os.path.isfile('zillow.csv'):
        
        # If csv file exists, read in data from csv file.
        df = pd.read_csv('zillow.csv', index_col=0)
        
    else:
        
        # Read fresh data from db into a DataFrame.
        df = new_zillow_data()
        
        # Write DataFrame to a csv file.
        df.to_csv('zillow.csv')
        
        
    return df

zillow_df = get_zillow_data()


zillow_df.head(1)

print("Total properties Shape: {}".format(zillow_df.shape))
print("-"*50)
zillow_df.info()


# =============================================================================
# Observations Based on the information above:
# =============================================================================
#Total number of single unit properties is 28060 
#Variable types distribution: 8 Float64 1 int64, 1 object

