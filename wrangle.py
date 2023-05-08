'''Wrangle Zillow Data'''

### IMPORTS ###

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from env import user,password,host
import env 

def check_file_exists(fn, query, url):
    if os.path.isfile(fn):
        print('csv file found and loaded\n')
        return pd.read_csv(fn, index_col=0)
    else: 
        print('creating df and exporting csv\n')
        df = pd.read_sql(query, url)
        df.to_csv(fn)
        return df 
    
def get_zillow_data():
    url = env.get_db_url('zillow')
    filename = 'zillow.csv'
    query = '''select 
                bedroomcnt
                , bathroomcnt
                , calculatedfinishedsquarefeet
                , taxvaluedollarcnt
                , yearbuilt
                , taxamount
                , fips
            from properties_2017
                join propertylandusetype
                    using (propertylandusetypeid)
            where propertylandusetypeid in (261, 279)
            '''

    df = check_file_exists(filename, query, url)

    return df 

def wrangle_zillow(df):
    df = df.rename(columns = {'bedroomcnt':'bedrooms',
                     'bathroomcnt':'bathrooms',
                     'calculatedfinishedsquarefeet':'area',
                     'taxvaluedollarcnt':'taxvalue',
                     'fips':'county'})
    
    df = df.dropna()
    
    make_ints = ['bedrooms','area','taxvalue','yearbuilt']

    for col in make_ints:
        df[col] = df[col].astype(int)
        
    df.county = df.county.map({6037:'LA',6059:'Orange',6111:'Ventura'})
            
    df = df [df.area < 25_000].copy()
    df = df[df.taxvalue < df.taxvalue.quantile(.95)].copy()
    
    return df


'''
### ACQUIRE DATA ###

def wrangle_zillow(user=user,password=password,host=host):
    """
    This function wrangles data from a SQL database of Zillow properties, caches it locally, drops null
    values, renames columns, maps county to fips, converts certain columns to integers, and handles
    outliers.
    
    :param user: The username for accessing the MySQL database
    :param password: The password is unique per user saved in env
    :param host: The host parameter is the address of the server where the Zillow database is hosted
    :return: The function `wrangle_zillow` is returning a cleaned and wrangled pandas DataFrame
    containing information on single family residential properties in Los Angeles, Orange, and Ventura
    counties, including the year built, number of bedrooms and bathrooms, square footage, tax value,
    property tax, and county. The DataFrame has been cleaned by dropping null values, renaming columns,
    mapping county codes to county names, converting certain columns
    """
    # name of cached csv
    filename = 'zillow.csv'
    # if cached data exist
    if os.path.isfile(filename):
        df = pd.read_csv(filename)
    # wrangle from sql db if not cached
    else:
        # read sql query into df
        # 261 is single family residential id
        df = pd.read_sql('select yearbuilt
                                    , bedroomcnt
                                    , bathroomcnt
                                    , calculatedfinishedsquarefeet
                                    , taxvaluedollarcnt
                                    , taxamount
                                    , fips 
                            from properties_2017
                            where propertylandusetypeid = 261'
                            , f'mysql+pymysql://{user}:{password}@{host}/zillow')
        # cache data locally
        df.to_csv(filename, index=False)
    # nulls account for less than 1% so dropping
    df = df.dropna()
    # rename columns
    df = df.rename(columns=({'yearbuilt':'year'
                            ,'bedroomcnt':'beds'
                            ,'bathroomcnt':'baths'
                            ,'calculatedfinishedsquarefeet':'area'
                            ,'taxvaluedollarcnt':'tax_value'
                            ,'taxamount':'prop_tax'
                            ,'fips':'county'}))
    # map county to fips
    df.county = df.county.map({6037:'LA',6059:'Orange',6111:'Ventura'})
    # make int
    ints = ['year','beds','area','tax_value']
    for i in ints:
        df[i] = df[i].astype(int)
    # handle outliers
    df = df[df.area < 25000].copy()
    df = df[df.tax_value < df.tax_value.quantile(.95)].copy()
    return df

'''
