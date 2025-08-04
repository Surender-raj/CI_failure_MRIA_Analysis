# The following code extracts data from Places api

import googlemaps
import time
import geopandas as gpd
from shapely.geometry import Polygon
from shapely.geometry import Point
from shapely.geometry import box
import numpy as np




# The following function extracts the list of places (e.g., chemical plants) using places nearby api

# The circles dataframe from the previous step (Step 3) with centre and radius (polygon defined) is one of the inputs
# The second input is the search string which is used to fetch the results.

# Finally, the function returns a dataframe with the list of all the places

def gmap_data_extractor(circles, search_string, api_key):
    
    # Using the personal api key to connect with google maps api
    gmaps = googlemaps.Client(key = api_key)

    list_of_plants = []
    
    for i in range(len(circles)):
        # This time command is essential to avoid 'too many requests' error from the api
        time.sleep(0.5)
        # The centre of the cirlce (lat,lon) is given in a tuple format
        loc = (circles.iat[i,0], circles.iat[i,1])
        # The following api command requires radius to be in metres. Hence, the radius in km is multiplied by 1000
        response = gmaps.places_nearby(location = loc , keyword = search_string , radius = circles.iat[i,2] * 1000)

        list_of_plants.extend(response.get('results'))

        # A page can hold only 20 results. If the next page token key is not a null, the below lines of code ensures
        # all the results are appended in the list
        next_page_token = response.get('next_page_token')
        
        while next_page_token:
            time.sleep(2)   # Important line Prevents Invalid Request
            response = gmaps.places_nearby(location = loc , keyword = search_string , radius = circles.iat[i,2] * 1000 , page_token = next_page_token)
            list_of_plants.extend(response.get('results'))
            next_page_token = response.get('next_page_token')
            
    df = gpd.GeoDataFrame(list_of_plants)
    return df

# The lat lon are stored in a nested dictionary. We entangle it by extracting the latitude and longitude from the geometry column
# Also the google api returns a viewport i.e., the co-ordinates of northeast and southwest location of the place
# We use this detail to create rectangular polygons and added as a geometry column to the df
# Also, a column containing centroid of the polygons is also added

def geometry_modifier(df): 
    latitude = np.zeros(len(df))
    longitude = np.zeros(len(df))
    bounds_polygon = []
    centroid = []

    for i in range(len(df)):
        latitude[i] = df.loc[i, 'geometry']['location']['lat']
        longitude[i] = df.loc[i, 'geometry']['location']['lng']
        
        centroid.append(Point(longitude[i] , latitude[i]))
        bounds_polygon.append(box(df.loc[i, 'geometry']['viewport']['southwest']['lng'],df.loc[i, 'geometry']['viewport']['southwest']['lat'],
                     df.loc[i, 'geometry']['viewport']['northeast']['lng'], df.loc[i, 'geometry']['viewport']['northeast']['lat']))
        
    df = gpd.GeoDataFrame(df)
    df = df.set_geometry(bounds_polygon)
    df.set_crs(epsg = '4326' , inplace = True)
    df['Centroid'] = centroid
    return df

def df_industrial_assets(nld_admin, circles, search_string, api_key):
    
    df = gmap_data_extractor(circles, search_string, api_key)
    
    # Modifies the geometry
    df1 = geometry_modifier(df)
    
    # Setting the CRS
    df1.set_crs(epsg = '4326' , inplace = True)
    
    # Drop the duplicates that come as a result of ovelap of circles
    df2 = df1.drop_duplicates(subset = ['geometry'])
    
    # Filter the dataframe for points within the region

    df3 = df2[df2.within(nld_admin.geometry.values[0])]
    df3 = df3.reset_index(drop = True)
    df4 = df3[['business_status' , 'geometry' , 'name']]
    return df4