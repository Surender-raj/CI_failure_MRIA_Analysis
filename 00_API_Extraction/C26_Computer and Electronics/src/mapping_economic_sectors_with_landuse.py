"""
Objective: To map the sites of economic activity. E.g. identifying the industrial sites wih chemical plants, steel fabricators etc.

"""


# Importing packages
import os
from input_loader import inputs
#from osmnx_extractor_polygon import osmnx_landuse_features
#from number_of_clusters_decision import optimum_number_of_clusters_method
import numpy as np
#from circle_creator import circles_search_request
import geopandas as gpd
import pandas as pd
from placesapiextract import df_industrial_assets
# from shapely.geometry import Polygon
# from shapely.geometry import Point


""""
Step1 : Reading the inputs
"""
# Input1: Economic sectors and the seach strings
# Input2: API key required to access Places API
# Input3: The shape file of the study region
# Input4: Clustreing inputs

# For detailed description of inputs refer documentation / input_loader.py

path_to_inputs = os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir)) , 'inputs')
filename = 'nuts2_zh.shp'

searchstrings, no_of_ss_per_sector, api_key, shp_admin, circles = inputs(path_to_inputs, filename)
print(searchstrings)


"""
Step4 : Extracting the data from Places API

""" 

# In this step, we create a dataframe that contains the asset locations corresponding to different economic sectors
# Refer sector_searchstrings.xlsx

# Initialising a geodataframe to store all the results
# The follwoing columns will be present in the dataframe
# a. business status b. geometry c. name d. Centroid e.Sector f. Sector_id

df_all_sectors_duplicates = gpd.GeoDataFrame()

for sector in range(len(searchstrings)):

    # Number of search strings for that sector
    no_of_strings = no_of_ss_per_sector[sector]

    for strings in range(no_of_strings):
        print(sector, strings)
        print(searchstrings.iloc[sector,1], searchstrings.iloc[sector,0])
        print(searchstrings.iloc[sector,3+strings])

        df = df_industrial_assets(shp_admin, circles, searchstrings.iloc[sector,3+strings], api_key)
        df['Sector'] = searchstrings.iloc[sector,1]
        df['Sector_id'] = searchstrings.iloc[sector,2]
        df['String'] = searchstrings.iloc[sector,3+strings]
        print(df.shape)

        df_all_sectors_duplicates  = gpd.GeoDataFrame(pd.concat([df_all_sectors_duplicates, df], ignore_index=True), geometry='geometry')

# Finally we drop the duplicates that arise because of common elements in different search strings       
df_all_sectors = df_all_sectors_duplicates.drop_duplicates(subset = ['geometry'])
df_all_sectors.Sector.value_counts()

df_all_sectors.to_file('C26.shp')