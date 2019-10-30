import pandas as pd
import numpy as np

def dropColumns(df, cols):
    """Dropping specified columns from the dataframe

       Parameters
       ----------

       df : A pandas dataframe with rows and columns

       cols: a column name as a string or multiple column names as a list of strings

       Returns
       -------

       df : A pandas dataframe wihout the dropped columns


    """
    return df.drop(cols, axis=1, inplace=True)

def transformData(df, df_test):
    dfx, df_testx = df.copy(), df_test.copy()
    
    dropColumns(dfx, 'neighbourhood') #1
    dropColumns(df_testx, 'neighbourhood')
    
    dfx = pd.get_dummies(dfx, columns=dfx.select_dtypes(include=['category']).columns, drop_first=False, prefix='dm') #2
    df_testx = pd.get_dummies(df_testx, columns=df_testx.select_dtypes(include=['category']).columns, drop_first=False, prefix='dm')
    
    dfx['log_price'] = np.log1p(dfx['price']) #3
    df_testx['log_price'] = np.log1p(df_testx['price'])
    
    dfx['log_minimum_nights'] = np.log1p(dfx['minimum_nights']) #4
    df_testx['log_minimum_nights'] = np.log1p(df_testx['minimum_nights']) 
    
    dfx['log_number_of_reviews'] = np.log1p(dfx['number_of_reviews']) #5
    df_testx['log_number_of_reviews'] = np.log1p(df_testx['number_of_reviews'])
    
    dfx['log_calculated_host_listings_count'] = np.log1p(dfx['calculated_host_listings_count']) #6
    df_testx['log_calculated_host_listings_count'] = np.log1p(df_testx['calculated_host_listings_count'])
    
    dfx['log_reviews_per_month'] = np.log1p(dfx['reviews_per_month']) #7
    df_testx['log_reviews_per_month'] = np.log1p(df_testx['reviews_per_month'])

    dfx['log_availability_365'] = np.log1p(dfx['availability_365']) #8
    df_testx['log_availability_365'] = np.log1p(df_testx['availability_365'])
        
    dfx['log_last_review'] = np.log1p(dfx['last_review']) #9
    df_testx['log_last_review'] = np.log1p(df_testx['last_review'])
    
    
    
    dropColumns(dfx, [
        'minimum_nights', 
        'number_of_reviews',
        'reviews_per_month', 
        'calculated_host_listings_count',
        'availability_365', 
        'last_review']) #10
    dropColumns(df_testx, [
        'minimum_nights', 
        'number_of_reviews',
        'reviews_per_month', 
        'calculated_host_listings_count',
        'availability_365', 
        'last_review'])
    
    return dfx, df_testx