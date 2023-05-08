import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
import numpy as np
import env
import os
 

""" This Function pulls in ght Telco_churn dataframe"""
def get_zillow_data():
    filename = "zillow.csv"
    if os.path.isfile(filename):
        return pd.read_csv(filename)
    else:
        url = env.get_db_url('zillow')
        # read the SQL query into a dataframe
        df = pd.read_sql('select bedroomcnt
	                        , bathroomcnt
                            , calculatedfinishedsquarefeet
                            , taxvaluedollarcnt 
	                        , yearbuilt
                            , taxamount
                            , fips
                            , propertylandusedesc
                            from properties_2017
                            	-- join propertylandusetype 
                        		-- using (propertylandusetypeid)
                                where propertylandusetypeid = 261
                                limit 100)

        # Write that dataframe to disk for later. Called "caching" the data for later.
        df.to_csv(filename)

        # Return the dataframe to the calling code
        return df 
