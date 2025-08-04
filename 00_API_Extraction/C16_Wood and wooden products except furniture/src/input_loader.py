# Importing pacakages

import pandas as pd
import geopandas as gpd
import os


def inputs(input_path, filename):
# Path to the inputs folder

    """ Input1"""

    # Input 1: The sectors and the search strings to be used in Places API
    # Creating a dataframe of the economic sectors and the search strings

    searchstrings = pd.read_excel(os.path.join(input_path, "sectors_searchstrings.xlsx"))

    # The total number of sectors of interest
    no_of_sectors = searchstrings.shape[0]

    # The maximum serach string per sector
    # This allows us to identify the maximum number of search strings to be used for any sector and will also help in further calculations 
    # number of search strings per sector
    max_searchstrings = searchstrings.shape[1]-3

    # Counting the number of NaN values in each row. Max serach strings - number of nan values in each row will help to identify the number 
    # number of search strings input for each sector

    nan_count = searchstrings.isna().sum(axis = 1)
    no_of_ss_per_sector = max_searchstrings - nan_count

    """ Input2"""

    # Input 2: 
    # # Reads the places API key
    api_key = open(os.path.join(input_path,'KEY.txt')).read()

    """ Input3"""

    # Input 3: Shape of the study area boundary.
    # A boundary of the study area in shape file format is expected in the input folder

    # Filter out only the shapefiles that ends with .shp
    files = os.listdir(input_path)

    
    shp_admin = gpd.read_file(os.path.join(input_path,filename))

    """ Input4""" 

    # Input 4: This set of circles to search the admin area
    
    circles = gpd.read_file(os.path.join(input_path,'circles_zh.shp'))


    return searchstrings, no_of_ss_per_sector, api_key, shp_admin, circles

