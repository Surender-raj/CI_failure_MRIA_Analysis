""" 
Author: Surender Raj Vanniya Perumal
Date: 11/02/2025
Purpose: To estimate the regional macroeconomic impacts of flood scenarions in Zuid Holland including substaion failures (for Paper 2)
Version: 1

"""

"""
Function blocks

"""

def depth_calculator(vector,raster, measure):
    
    # CRS consistency between the vector and the raster
    vector = vector.to_crs(raster.rio.crs)
    
    # Converting no data (i.e., no inundationvalues) to zero
    raster = raster.where(raster != raster.rio.nodata, 0)
    
    # Calucate the maximum inundation depth within each polygon
    stats = zonal_stats(vector, raster[0].values, affine=raster.rio.transform(), stats= measure)
    
    # Create a new column depth and if the stat is None (i.e., the polygon is outside the raster extent) we make the depth zero
    vector['depth'] = [stat[measure] if stat[measure] is not None else 0 for stat in stats]
    
    return vector

def mria_inputs(input_path):

    # datapath to the inputs folder
    data_path = input_path

    # file path to SUT tables
    filepath = os.path.join(data_path,'MRIO', 'mria_nl_sut.xlsx')
    mria = pd.read_excel(filepath, sheet_name = 'SUP', header = [0,1], index_col = [0,1])
    regions_ineu = ((mria.index.get_level_values(0)).unique()).tolist()

    # Unique regions in the SUT table
    regions = regions_ineu


    # name of the table
    name = 'nl_sut'

    # Preparing the data in the MRIA model format
    DATA = sut_basic('nl_sut', filepath,regions)
    DATA.prep_data()
    data_source = 'nl_sut'

    return DATA, regions

def sort_df(gdf_polygon, gdf_points):
    nearest_points_series = gdf_points.geometry.apply(lambda x: gdf_polygon.geometry.iloc[0].distance(x))
    sorted_points_df = gdf_points.assign(distance_to_polygon=nearest_points_series).sort_values(by='distance_to_polygon')
    sorted_points_df['rank'] = range(1, len(sorted_points_df) + 1)
    return sorted_points_df

def ci_assignment(exposure, substations, p, rarray):

    # Conevrting the substtions to a points dataframe
    ss_pts = substations
    ss_pts['geometry'] = substations['geometry'].centroid
    
    for sr in range(len(exposure)):
        soi = exposure.iloc[sr]
        soi = soi.to_frame().T
        soi = soi.reset_index(drop = True)

        # Sorting the substations based on closest proximity to the site
        sorted_df = sort_df(soi, ss_pts)
        sorted_df = sorted_df.reset_index(drop = True)

        for i in range(len(sorted_df)):
            rank = sorted_df.loc[i,'rank']
            prob_new = ((1-p)**(rank-1) ) * p
            sorted_df.loc[i,'probability'] = prob_new

        # Multiplying probability with voltage as weights
        sorted_df['probxweight'] = sorted_df['probability']*1

        # Calculating weighted probabilities
        sorted_df['weigh_prob'] = sorted_df['probxweight']/sorted_df['probxweight'].sum()

        # Deriving cumulative probabilities
        sorted_df.loc[0, 'cum_prob'] = sorted_df.loc[0, 'weigh_prob']
        
        for i in range(len(sorted_df)-1):
            sorted_df.loc[i+1, 'cum_prob'] = sorted_df.loc[i+1, 'weigh_prob'] + sorted_df.loc[i, 'cum_prob']

        # Selecting a substation based on a uniformly distributed random number 
        r = rarray[sr]
        selected_row = sorted_df[sorted_df['cum_prob'] >= r].iloc[0]
        exposure.loc[sr,'substation_osm'] = selected_row.osmid
       
    return exposure

def functionality_modifier(exposure, substations):

    for i in range(len(exposure)): 
        
        # Filtering the expsoure site of interest
        soi = exposure.iloc[i]
        soi = soi.to_frame().T
        soi = soi.reset_index(drop = True)
        
        # Osm id of the connected substation
        osm_ss = soi.loc[0,'substation_osm']

        # Functionality state of the substation
        ss_i  = substations[substations['osmid'] == osm_ss]
        ss_i = ss_i.reset_index(drop = True)
        exposure.loc[i,'power_fail'] = ss_i.loc[0,'flood_fail']
            
    return exposure

def sector_level_production(sectors, osm_industrial):
    sectors = sectors.to_crs(osm_industrial.crs)
    tot_sectors = sectors.Sector_id.unique()
    prod_df = pd.DataFrame(columns = ['sector', 'wo_ci', 'wi_ci'])
    
    for i in range(len(tot_sectors)):
        sector_filt = sectors[sectors['Sector_id'] == tot_sectors[i]]
        
        # Calculating centroid and setting it as the active geometry column
        sector_filt["centroid"] = sector_filt.geometry.centroid  
        sector_filt = sector_filt.set_geometry("centroid")  # Temporarily set centroids as geometry

        # Mapping the columns of osm industrial with sector specific assets
        # The sites that are not wihtin the boundaries of the industrial sites from OSM are considered insginifcant in production and excluded from the analysis
        warnings.filterwarnings("ignore")
        sector_filt = gpd.sjoin(sector_filt, osm_industrial[["osmid",'geometry', 'flood_fail','functional']], how="inner", predicate="within")
        sector_filt["area"] = sector_filt["osmid"].map(osm_industrial.set_index("osmid").geometry.area)

        sector_filt['flood_fail_area'] = sector_filt['flood_fail'] * sector_filt['area']
        sector_filt['functional_area'] = sector_filt['functional'] * sector_filt['area']

        # production level without CI
        prod_woci = sector_filt['flood_fail_area'].sum() / sector_filt['area'].sum()

        # production level with CI
        prod_wici = sector_filt['functional_area'].sum() / sector_filt['area'].sum()

        prod_df.loc[i,'sector'] = tot_sectors[i]
        prod_df.loc[i,'wo_ci'] = prod_woci
        prod_df.loc[i,'wi_ci'] =  prod_wici

    return prod_df

def create_distance_dict(nl_nuts, beta, regions):
    """

    Step 3: Creating a distance dictionary to limit disatser imports
    Here, it is a redundant since we donot take into the effecte of geographical distance
    The parameter beta value is set to zero

    """

    nl_nuts_crs = nl_nuts.to_crs(epsg = 3857)
    nl_nuts_crs['centroid'] = nl_nuts_crs.centroid

    distance_dict = {}

    for i in range(len(regions)):
        for j in range(len(regions)):

            r1 = regions[i]
            r2 = regions[j]
            pair = (r1,r2)

            df1 = nl_nuts_crs[nl_nuts_crs['NUTS_ID'] == r1]
            df2 = nl_nuts_crs[nl_nuts_crs['NUTS_ID'] == r2]

            # Centroid
            centroid1 = df1['geometry'].centroid.iloc[0]  
            centroid2 = df2['geometry'].centroid.iloc[0]  

            # Distance in 100s of km
            distance_km = centroid1.distance(centroid2) / 100000
            inv_distance = 1/((distance_km+0.01)**beta)
            distance_dict[pair] = min(1,inv_distance)

    return distance_dict

def mria_todf(data):
    df = pd.DataFrame(list(data.items()), columns=['index', 'value'])
    df[['Index1', 'Index2']] = pd.DataFrame(df['index'].tolist(), index=df.index)
    df = df.drop(columns=['index'])
    df = df.set_index(['Index1', 'Index2'])
    return df

def mria_todf1(data):
    df = pd.DataFrame(list(data.items()), columns=['index', 'value'])
    df[['Index1', 'Index2', 'Index3']] = pd.DataFrame(df['index'].tolist(), index=df.index)
    df = df.drop(columns=['index'])
    df = df.set_index(['Index1', 'Index2', 'Index3'])
    return df

def ineff_calculator(MRIA_RUN):

    supply_values_wimp = {(i, j): value(MRIA_RUN.product_supply[i, j]) for i in MRIA_RUN.m.r for j in MRIA_RUN.m.P}
    sup_wimp = mria_todf(supply_values_wimp)

    # Product demand
    dem_values_wimp = {(i, j): value(MRIA_RUN.product_demand[i, j]) for i in MRIA_RUN.m.r for j in MRIA_RUN.m.P}
    dem_wimp = mria_todf(dem_values_wimp)
    
    inefficiency = sup_wimp - dem_wimp
    inefficiency = inefficiency.unstack(level = 0)
    return inefficiency

def results_printer(MRIA_RUN1, MRIA_RUN2, MRIA_RUN3, MRIA_RUN5, DATA, op_factor, imp_flex, solvername, extension):

    #All outputs

    Xdis1 = mria_todf(MRIA_RUN1.X.get_values())
    Xdis1 = Xdis1.unstack(level = 0)
    Xdis1.to_excel(os.path.join('results', f'Xdis1_{op_factor}_{imp_flex}_{solvername}_{extension}.xlsx'))


    Xdis2 = mria_todf(MRIA_RUN2.Xdis.get_values())
    Xdis2 = Xdis2.unstack(level = 0)
    Xdis2.to_excel(os.path.join('results', f'Xdis2_{op_factor}_{imp_flex}__{solvername}_{extension}.xlsx'))


    Xdis3 = mria_todf(MRIA_RUN3.Xdis.get_values())
    Xdis3 = Xdis3.unstack(level = 0)
    Xdis3.to_excel(os.path.join('results', f'Xdis3_{op_factor}_{imp_flex}_{solvername}_{extension}.xlsx'))


    Xdis5 = mria_todf(MRIA_RUN5.X.get_values())
    Xdis5 = Xdis5.unstack(level = 0)
    Xdis5.to_excel(os.path.join('results', f'Xdis5_{op_factor}_{imp_flex}__{solvername}_{extension}.xlsx'))


    # Rationing

    Ddis = mria_todf(MRIA_RUN3.Ddis.get_values())
    Rat = Ddis.unstack(level = 0)
    Rat.to_excel(os.path.join('results', f'Rat_{op_factor}_{imp_flex}_{solvername}_{extension}.xlsx'))


    # Disaster imports

    Dimp2 = mria_todf1(MRIA_RUN2.disimp.get_values())
    Dimp2.to_excel(os.path.join('results', f'Dimp2_{op_factor}_{imp_flex}_{solvername}_{extension}.xlsx'))

    Dimp3 = mria_todf1(MRIA_RUN3.disimp.get_values())
    Dimp3.to_excel(os.path.join('results', f'Dimp3_{op_factor}_{imp_flex}_{solvername}_{extension}.xlsx'))

    # Xbase
    Xbase_ini = {(i, j): value(MRIA_RUN3.Xbase[i, j]) for i in MRIA_RUN1.m.r for j in MRIA_RUN1.m.S}
    Xbase = mria_todf(Xbase_ini)
    Xbase = Xbase.unstack(level = 0)
    Xbase.to_excel(os.path.join('results', f'Xbase_{op_factor}_{imp_flex}_{solvername}_{extension}.xlsx'))

    # Value Added inital

    VA_ini = {(i, j): value(DATA.ValueA[i, j, 'Imports']) for i in MRIA_RUN3.m.r for j in MRIA_RUN1.m.S}
    VA_df = mria_todf(VA_ini)
    VA_df = VA_df.unstack(level = 0)
    VA_df.to_excel(os.path.join('results', f'VA_{op_factor}_{imp_flex}_{solvername}_{extension}.xlsx'))


    ineff2 = ineff_calculator(MRIA_RUN2)
    ineff2.to_excel(os.path.join('results', f'ineff2_{op_factor}_{imp_flex}_{solvername}_{extension}.xlsx'))

    ineff3 = ineff_calculator(MRIA_RUN3)
    ineff3.to_excel(os.path.join('results', f'ineff3_{op_factor}_{imp_flex}_{solvername}_{extension}.xlsx'))

    ineff5 = ineff_calculator(MRIA_RUN5)
    ineff5.to_excel(os.path.join('results', f'ineff5_{op_factor}_{imp_flex}_{solvername}_{extension}.xlsx'))


"""
Step 1:Importing necessary packages and MRIA code files

"""

import rioxarray as rio
import geopandas as gpd
from rasterstats import zonal_stats
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
import os
from table import sut_basic
from run_mria import mria_run
from pyomo.environ import value
import math

# Combining them in a function


def macroimpact_estimation(RP1,op_factor1, imp_flex1, p1, rarray, m):

    """
    Step 2: User inputs (check before running the analysis)

    """

    # Return period of the flood map
    RP = RP1

    # Raster stats can estimate the mean and maximum depth within the polygons
    # The final results are sensitive to the metric chosen 

    meanmaxci = 'max'  # CIs are usually smaller and maximum flood depth could be a guiding factor
    meanmaxin = 'mean' # Inudstrial sites are usually larger

    # Damage threshold for CI and industrial sites. Again, the reuslts are sensitive to the parameters chosen

    dam_thres_ci = 1 # flood depth in meters
    dam_thres_in = 1   # flodo depth in meters

    # Geometric distribution spread parameter p
    # p = 1 connects an industrial sites to the nearest substaion p less than 1 increase the spread and 
    p = p1


    # MRIA PARAMETERS OF THE MODEL

    #Parameter  to determine the steepness of distance function
    #Creating a distance dictionary to limit disatser imports distant countries
    #Here, it is a redundant since we donot take into the effecte of geographical distance since the analysis is country wide
    #The parameter beta value is set to zero

    beta = 0 

    # Overproduction factor
    op_factor = op_factor1

    # Trade flexibility factor 
    imp_flex = imp_flex1


    # Solver to use  (options: 'mosek', 'gams')
    # GAMS uses conopt

    solvername = 'mosek'

    # Regional code of Zuid-Holland
    roi = 'NL33'

    """
    Step 3: Data loader

    """

    # Path to the data folder
    input_path = os.path.join(os.path.dirname(os.getcwd()), 'data')

    # Input 1: Hazard file
    floodmap = rio.open_rasterio(os.path.join(input_path,f'rotterdam_event_RP{RP}.tif'))

    # Input 2: Critical Infrastrcuture data - Substations
    ci = gpd.read_parquet(os.path.join(input_path, 'substaions_nl33.parquet'))

    # Input 3: Industrial sites from Openstreet Maps
    osm_industrial = gpd.read_parquet(os.path.join(input_path, 'is_nl33.parquet'))

    # Input 4: Economic sector specific site locations
    sectors = gpd.read_parquet(os.path.join(input_path, 'industrial.parquet'))

    # Input 5: Suply and Use table for Netherlands
    DATA, regions = mria_inputs(input_path)

    # Input 6: NUTS-2 shape file of Netherlands
    nl_nuts = gpd.read_file(os.path.join(input_path, 'nl_nuts.shp'))


    """
    Step 4: Estimating flood depths at CI and OSM industrial site locations

    """

    ci = depth_calculator(ci,floodmap, measure = meanmaxci)
    osm_industrial = depth_calculator(osm_industrial,floodmap, measure = meanmaxin)

    """
    Step 5: Assigning damage tags based on direct flood exposure 0: flooded 1: non-flooded

    """

    ci["flood_fail"] = np.where(ci["depth"] > dam_thres_ci, 0, 1)
    osm_industrial["flood_fail"] = np.where(osm_industrial["depth"] > dam_thres_in, 0, 1)


    """
    Step 6: Assigning substations to the industrial sites based on a geometric distribution
            The functional state of the site now also depends on if the substations is functional or not

    """

    # Assigns substation based on the geometric distribution
    osm_industrial = ci_assignment(osm_industrial, ci, p, rarray)

    # Creates a new functional column with subtstaion failure incorporated
    osm_industrial = functionality_modifier(osm_industrial, ci)
    osm_industrial['functional'] = osm_industrial['flood_fail'] * osm_industrial['power_fail']


    """
    Step 7: Linking industrial sites to production decline in sectors

    """

    prod_df = sector_level_production(sectors, osm_industrial)
    dict_wo_ci = {(roi, sector): value for sector, value in zip(prod_df['sector'], prod_df['wo_ci'])}
    dict_wi_ci = {(roi, sector): value for sector, value in zip(prod_df['sector'], prod_df['wi_ci'])}


    dict_wi_ci

    """
    Step 8: MRIA_RUN

    """

    # see comments near beta parametre step 2
    distance_dict = create_distance_dict(nl_nuts, beta, regions)

    # Without CI
    #MRIA_RUN1_NOCI, MRIA_RUN2_NOCI, MRIA_RUN3_NOCI, MRIA_RUN5_NOCI = mria_run(DATA, op_factor,1, imp_flex, dict_wo_ci , {}, distance_dict, solvername)

    # With CI
    MRIA_RUN1_CI, MRIA_RUN2_CI, MRIA_RUN3_CI, MRIA_RUN5_CI = mria_run(DATA, op_factor,1, imp_flex, dict_wi_ci , {}, distance_dict, solvername)

    # System impact results
    #results_printer( MRIA_RUN1_NOCI, MRIA_RUN2_NOCI, MRIA_RUN3_NOCI, MRIA_RUN5_NOCI, DATA, op_factor, imp_flex, solvername, extension = f'noci_{RP}')
    results_printer( MRIA_RUN1_CI, MRIA_RUN2_CI, MRIA_RUN3_CI, MRIA_RUN5_CI, DATA, op_factor, imp_flex, solvername, extension = f'ci_{RP}')

    # Damage and funcitonality analysis results
    ci.to_excel(os.path.join('results', f'ci_{op_factor}_{imp_flex}_{solvername}_{RP}.xlsx'))
    osm_industrial.to_excel(os.path.join('results', f'osm_industrial_{op_factor}_{imp_flex}_{solvername}_{RP}_{p1}_{m}.xlsx'))
    prod_df.to_excel(os.path.join('results', f'prod_df_{op_factor}_{imp_flex}_{solvername}_{RP}.xlsx'))

# Calculating expected annual cost
def EAC(df):
    df['p'] = 1 / df.index
    zero_row = pd.DataFrame({'p': [1], 'cost': [0]})
    df = pd.concat([zero_row, df], ignore_index=True)
    df = df.sort_values('p')
    area = np.trapz(df['cost'].values, x=df['p'].values)
    return area

def expected_annual_impact_calculation(op1, if1, cif, RP, k, m):

    total_cost_results = pd.DataFrame()
    for i in range(len(RP)):

            rp = RP[i]
            ci_f = cif
            op_factor = op1
            imp_flex  = if1
            path = os.path.join(os.getcwd(), 'results')
            
            # Rationing corrected output loss
            Xbase = pd.read_excel(os.path.join(path, f'Xbase_{op_factor}_{imp_flex}_mosek_{ci_f}_{rp}.xlsx'), index_col = [0], header = [1])
            Xbase = Xbase.drop("Index2", axis = 'index')
            Xbase_sum = Xbase.sum().sum()
            
            X1 = pd.read_excel(os.path.join(path, f'Xdis3_{op_factor}_{imp_flex}_mosek_{ci_f}_{rp}.xlsx'), index_col = [0], header = [1])
            X1 = X1.drop("Index2", axis = 'index')
            X1_sum = X1.sum().sum()
            
            rat = pd.read_excel(os.path.join(path, f'Rat_{op_factor}_{imp_flex}_mosek_{ci_f}_{rp}.xlsx'), index_col = [0], header = [0,1])
            rat_sum = rat.sum().sum()
            
            c3 = (Xbase_sum - X1_sum) - rat_sum
            
            
            # Actual inefficiency
            ineff1 = pd.read_excel(os.path.join(path, f'ineff3_{op_factor}_{imp_flex}_mosek_{ci_f}_{rp}.xlsx'), index_col = [0], header = [1])
            ineff1 = ineff1.drop("Index2", axis = 'index')
            ineff1 = ineff1.sum().sum()
            
            c2 = ineff1
            
            # Production equivalent of ratioing
            X2 = pd.read_excel(os.path.join(path, f'Xdis5_{op_factor}_{imp_flex}__mosek_{ci_f}_{rp}.xlsx'), index_col = [0], header = [1])
            X2 = X2.drop("Index2", axis = 'index')
            X2_sum = X2.sum().sum()
            c1 = X2_sum 

            cost = (c1 + c2+ c3 ) / 365
            total_cost_results.loc[rp, 'cost_per_day'] = cost

    recovery_duration = []
    no_of_flooded = []

    for i in range(len(RP)):
        rp = RP[i]
        path = os.path.join(os.getcwd(), 'results')
        osm = pd.read_excel(os.path.join(path, f'osm_industrial_{op_factor}_{imp_flex}_mosek_{rp}_{k}_{m}.xlsx'), index_col = [0], header = [0])
        osm_failed = osm[osm['flood_fail'] == 0]
        osm_failed['duration'] = (osm_failed['depth'] * 16.6) + 7.738  # From T. Endendijk et al.(2024)
        recovery_duration.append(osm_failed['duration'].mean())
        no_of_flooded.append(len(osm_failed))
        
    # This is added as a scaling factor to the average recovery duration to accomodtae for the fact that more number of sites will take longer time due to resource constraints and vice vers
    no_of_flooded_log = [math.log10(x+1) for x in no_of_flooded]
    total_cost_results['cost'] = total_cost_results['cost_per_day'] * recovery_duration * no_of_flooded_log
    eac = EAC(total_cost_results)
    return eac


# Analysis 

RPs = [10, 100, 1000, 10000]    # Return periods
OPs = [1.01]                # Overproduction factor
IFs = [0.25]                # Flexibility factor

# Calculating the number of random numbers required
input_path = os.path.join(os.path.dirname(os.getcwd()), 'data')
osm_industrial = gpd.read_parquet(os.path.join(input_path, 'is_nl33.parquet'))
rand_len = len(osm_industrial)


eac_df = pd.DataFrame()

# Simulation parameters
sparseness = [0.8, 0.6, 0.4, 0.2, 0.01]


nos = 50  # Number of simulations

for m in range(nos):

    rarray = np.random.rand(rand_len)

    for k in range(len(sparseness)):
        spar = sparseness[k]

        for i in range(len(OPs)):
            for j in range(len(RPs)):

                rp = RPs[j]
                op = OPs[i]
                tf = IFs[i]

                macroimpact_estimation(rp, op, tf, spar, rarray, m)

        
                
            eac_ci = expected_annual_impact_calculation(op, tf, 'ci', RPs, spar , m)
            eac_df.loc[m,spar] = eac_ci
        
        print(m,k)

eac_df.to_excel('eac_df_sparseness.xlsx')


