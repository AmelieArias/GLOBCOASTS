#!/usr/bin/env python
# coding: utf-8

# In[5]:


print("#################################")
print("# - Importing modules...")

import math
from math import *
import datetime
from datetime import *
import time
import sys
import os
import cmcrameri
from pyproj import Geod
import statsmodels.api as sm
from cmocean import cm

print("#################################")
print("# - Importing modules...")

try:
    print("# - Importing numpy")
    import numpy as np
    from numpy import *
except:
    raise ImportError(" ERROR importing numpy")
try:
    print("# - Importing pandas")
    import pandas as pd
except:
    raise ImportError(" ERROR importing pandas")
    
try:
    import geopandas as gpd
except:
    raise ImportError(" ERROR importing geopandas")
    
try:
    print("# - Importing pylab")
    import pylab
    from pylab import *
except:
    raise ImportError(" ERROR importing pylab")
try:
    print("# - Importing matplotlib")
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors
    import matplotlib.gridspec as gridspec
    import matplotlib.patches as mpatches
    import matplotlib.lines as mlines
    from matplotlib.gridspec import GridSpec
    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
    import matplotlib.font_manager as fm
    from cartopy import crs as ccrs, feature as cfeature
    from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
    from cartopy.mpl.gridliner import LongitudeFormatter, LatitudeFormatter
    from matplotlib.path import Path
    from cartopy.mpl.patch import geos_to_path
except:
    raise ImportError(" ERROR importing matplotlib")
try:
    print("# - Importing scipy")
    import scipy as sc
    from scipy import interpolate
    from scipy.interpolate import interp1d, interp2d
    from scipy.interpolate import griddata
    from scipy.stats import *
    import scipy.io as mio
    from scipy.spatial import distance
    from scipy.ndimage import gaussian_filter1d
    from scipy.optimize import minimize
    from scipy.optimize import minimize_scalar

except:
    raise ImportError(" ERROR importing scipy")
try:
    print("# - Importing gc")
    import gc
except:
    raise ImportError("  ERROR importing gc")
try:
    print("# - Importing socket")
    import socket
except:
    raise ImportError("  ERROR importing socket")
try:
    print("# - Importing getpass")
    import getpass
except:
    raise ImportError("  ERROR importing getpass")
try:
    import configparser
    from configparser import *
except:
    raise ImportError("  ERROR importing ConfigParser")
try:
    print("# - Importing statistics")
    import statistics
except:
    raise ImportError("  ERROR importing statistics")

try:
    print("# - Importing h5py")
    import h5py
except:
    raise ImportError("  ERROR importing h5py")
try:
    print("# - Importing xarray")
    import xarray as xr
except:
        raise ImportError("  ERROR importing xarray")
        
try:
    print("# - Importing cartopy")
    import cartopy
except:
    raise ImportError(" ERROR importing cartopy")

try:
    print("# - Importing globe")
    from global_land_mask import globe
except:
    raise ImportError(" ERROR importing globe")
    
try:
    print("# - Importing netcdf4")
    from netCDF4 import Dataset
except:
    raise ImportError(" ERROR importing netcdf")
    
from tqdm import tqdm
from scipy.signal import butter, filtfilt


# In[3]:


# PATH DEFINITION TO TOOLBOXES DIRECTORY => inputs and functions
# CHANGE WITH YOUR FILE DIRECTORY

pathin  = 'C:/Users/arias/ownCloud/GLOBCOAST/'
pathout = 'C:/Users/arias/ownCloud/GLOBCOAST/RESULTS/'

try:
    print("# - Importing main toolboxes and input path...")
    global PTOOL
    # If path to TOOLBOXES is in you environment variables
    PTOOL = pathin + 'FUNCTION/' 
    INPUT = pathin + 'INPUT/' 
except:
    raise ImportError(" ERROR path is not defined ")

# IMPORT OF TOOLBOXES
# (must respect dependancies between toolboxes)

print("# - Importing personal toolboxes...")
try:
    sys.path.append(PTOOL)
except:
    raise ImportError(" ERROR adding TOOLBOX PATH")
    
#  MAIN TOOLBOX
try:
    import function_GLOBCOASTS as FUNC    # toolbox where are all the functions I create for the model
except:
    raise ImportError(" ERROR importing FUNC")

    
print("# - Toolboxes are OK")


# In[6]:


# INPUT IMPORT
# ---------------------------------------------------------------------------------------
print("# - Uploading files ...")
# ---------------------------------------------------------------------------------------
# SOLID RIVER DISCHARGE CALCULATED THROUGH BQART FORMULA WITH Te = 0.2 [Mt/yr]
BQART_brut   = np.loadtxt(INPUT+ 'Sorties_Bqart.txt', delimiter=';')
BQART_m3     = BQART_brut[:,0][:8841]/2.650 #to convert it to m3
BQART        = BQART_m3/12            # m3/yr --> m3/month

# ---------------------------------------------------------------------------------------
# SEADATAS OBTAIN GLOBALLY WITH ERA5 REANALYSIS (Hs,Tp,Dir), ISBA-CTRIP (River discharge), MOG2D(DAC) and CMEMS(SLA)
SEADATA = xr.open_dataset(INPUT + 'SEADATA_14140pts_1993_2019-analysed.nc',engine='netcdf4')

SEADATA_lon = SEADATA['lon'].values #[°]
SEADATA_lat = SEADATA['lat'].values

SEADATA_Hs  = SEADATA['Hs_mounthly'].values     #m
SEADATA_Tp  = SEADATA['Tp_mounthly'].values     #s
SEADATA_Dir = SEADATA['dir_mounthly'].values    #°N
SEADATA_SLA = SEADATA['sla_detrend'].values     #m
SEADATA_DAC = SEADATA['dac_detrend'].values     #m
SEADATA_RIV = SEADATA['rivdis_mounthly'].values #m3/month

# INDEX SELECTION TO MATCH OUR STUDY AREA
lon         = SEADATA_lon[:8841]
lat         = SEADATA_lat[:8841]
Hs          = SEADATA_Hs[84:,:8841]
Tp          = SEADATA_Tp[84:,:8841]
Dir         = SEADATA_Dir[84:,:8841]
SLA         = SEADATA_SLA[84:,:8841]
DAC         = SEADATA_DAC[84:,:8841]
RiverD      = SEADATA_RIV[84:,:8841]

# ---------------------------------------------------------------------------------------
# TIDE INPUT TO COMPUTE SLOPE THROUGH SUNAMURA 84 MODIFIED FORMULA (Arias et al., 2025)
TIDE        = sc.io.loadmat(INPUT + 'Tide_Glob.mat')
Tide_range  = TIDE['Tide_max'][0][:8841] #m

# ---------------------------------------------------------------------------------------
# DYNAMICAL DEPTH OF CLOSURE (Young, 1995)
Dc        = xr.open_dataset(INPUT + 'Dc.nc',engine='netcdf4')
DoC       = Dc['Dc'].values #m

# ---------------------------------------------------------------------------------------
# WORLDWIDE CALIBRATION COEFFICIENT

# ---------------------------------------------------------------------------------------
# VALDIATION FILE : WATERLINE POSITION OBTAIN THANKS TO LANDSAT 7 AND 8 PRODUCTS
Validation  = sc.io.loadmat(INPUT + 'Shorelines_global_20231101_shift.mat')
Xshores_val = Validation['X_safe'][:,108:-12]
latX        = Validation['latX'][:]
lonX        = Validation['lonX'][:]

Xshores_VALIDATION = np.transpose((-1*Xshores_val))

print("# - All files are upload...")


# In[7]:


# CONSTANT VALUES DECLARATION
print("# - Initializing constant values...")
# Median grain size [m]
d50 = 10e-3

# Sand porosity
poro = 0.4

# Sand density [kg/m3]
rohs      = 2650 

# Water density [kg/m3]
roh       = 1000 

# gravitationnal acceleration [m/s2]
g = 9.81

# Earth radius [m]
R = 6371000

# Number of time step
Nmonth = len(SLA)


# In[8]:


# INPUT CALCULATION 
# ---------------------------------------------------------------------------------------
# TIDE CHECK FOR NAN VALUES
for i in range(1,len(Tide_range)):
        if str(Tide_range[i]) == 'nan':
            Tide_range[i]=Tide_range[i-1]
            
print("# - Tide range is cleaned")
 
# ---------------------------------------------------------------------------------------
# SOLID RIVER DISCHARGE VARIABILITY THROUGH THE USE OF ISBA-CTRIP DATASET 
# 1. CLEAN LIQUID RIVER DISCHARGE DATASET (ISBA-CTRIP)
for line in RiverD:
    for i in range(len(line)):
        if str(line[i]) == 'nan':
            line[i] = 0
            
# 2. SOLID RIVER DISCHARGE VARIBILITY CALCUL
QrivD = FUNC.RIVDIS(RiverD,BQART)
for line in QrivD:
    if str(line) == 'nan':
        print(line)
print("# - QrivD is ok")
            
            
# ---------------------------------------------------------------------------------------
# TOTAL WATER LEVEL CALCUL
# 1. Foreshore Beach slope calculation through SUNAMURA 84 modified by tide (Arias et al., 2025)

L_SU = (g/(2*np.pi))*Tp**2                                                       # wave length

beta       = 0.12 * ((np.sqrt(2*np.pi*d50*L_SU))/(Hs * (1+(Tide_range/Hs))))**(1/2) #slope

for i in range(len(beta)):
    for j in range(len(beta[0])):
        if str(beta[i,j]) == 'nan':
            print(i,j)
print("# - Foreshore slopes is ok ")

# 2. Wave set up (Stockdon,2006 ) -> Approximation of a linear slope
SU   = 0.35*beta*np.sqrt(Hs*L_SU) #[m]
print("# - Set up is ok")

#3. Sum of sea level anomaly (SLA), storm surge (DAC) and wave set-up (SU)
TWL  = SLA + DAC+ SU

print("# - Total water level is ok")


# In[9]:


# LAT/LON PRE-TREATMENT TO GET RIDE OF SATELLITE "JUMPS"
# ---------------------------------------------------------------------------------------
#1. Search for the index of position spaced by less than 0.55° to determine the zones (0.55 is the max resolution of our dataset)
start_index, end_index = FUNC.find_index(lon, lat,0.55)

#2. Zone creation through the fusion of lat/lon 
join_section,join_index = FUNC.join_sections(start_index, end_index, lon, lat,0.55)

#3. Selection of the zone that have more than 2 position
join_section_filtered = [(i, j) for i, j in join_index if abs(i - j) > 2]


# In[10]:


# MAIN CODE
# ---------------------------------------------------------------------------------------
dt = 1                            # time step 1 month

# ---------------------------------------------------------------------------------------
# LIST CREATION FOR OUTPUTS
# 1. INPUTS AND CALCULATED VARIABLES
r_index              = []
r_dKAMP              = []                    # Spatial difference of Longhsore transport rate
r_dTWL               = []                    # Time difference of total water level
r_TWL                = []                    # total water level
r_Ls                 = []                    # Distance between two coastal point
r_alpha              = []                    # Azimuth angle (coastline orientation °N)
r_normal             = []                    # Coastline normal angle
r_incidence_angle    = []                    # Local incidence angle between the waves and the coastline normal

# 2. CROSS-SHORE POSITION DELTA (m/month)
r_dx_CS_Hydro        = []                    # delta of position due to HYDROLOGICAL COMPONENT
r_dx_CS_MorphoLST    = []                    # delta of position due to the SEDIMENTARY BUDGET (LST + River) COMPONENT
r_dx_CS_MorphoXshore = []                    # delta of position due to the SLOPE VARIATION COMPONENT
r_dx_CS_MorphoTOT    = []                    # delta of position due to the TOTAL MORPHOLOGICAL COMPONENT
r_dx_CS_TOTAL        = []                    # TOTAL delta position 

# 3. CROSS-SHORE POSITION VARIABILITY (m - X + dX for each time step)
r_X_CS_MorphoLST     = []
r_X_CS_MorphoTOT     = []
r_X_CS_Hydro         = []
r_X_CS_MorphoXshore  = []
r_X_CS_TOTAL         = []

# 4. RESULTING GEOGRAPHICAL LON/LAT
r_Lon                = []                    # LONGITUDE °
r_Lat                = []                    # LATITUDE °
r_Lon_m              = []                    # LONGITUDE m
r_Lat_m              = []                    # LATITUDE m
r_DLon               = []                    # LONGITUDE DIFFERENCE BETWEEN TWO CONSECUTIVE COASTPOINT m
r_DLat               = []                    # LATITUDE DIFFERENCE BETWEEN TWO CONSECUTIVE COASTPOINT m
r_dLon               = []                    # LONGITUDE DIFFERENCE BETWEEN TWO CONSECUTIVE COASTPOINT °
r_dLat               = []                    # LATITUDE DIFFERENCE BETWEEN TWO CONSECUTIVE COASTPOINT °


# In[11]:


# 1. FIRST SPATIAL LOOP WHICH SELECT EACH JOIN SECTION INDEPENDENTLY
for i, section in tqdm(enumerate(join_section_filtered)):
    start_index                 = section[0]
    end_index                   = section[1]

    index                       = np.arange(start_index, end_index)
    zone_length                 = len(index)

    # a. INITIALISATION OF RESULTS TABLE FOR THE SECTION
    dKAMP                       = np.zeros((Nmonth, zone_length))
    dTWL                        = np.zeros((Nmonth, zone_length))
    TWL                         = np.zeros((Nmonth, zone_length))
    Ls                          = np.zeros((Nmonth, zone_length))
    alpha                       = np.zeros((Nmonth, zone_length))
    normal                      = np.zeros((Nmonth, zone_length))
    incidence_angle             = np.zeros((Nmonth, zone_length))

    dx_CS_Hydro                 = np.zeros((Nmonth, zone_length))
    dx_CS_MorphoLST             = np.zeros((Nmonth, zone_length))
    dx_CS_MorphoTOT             = np.zeros((Nmonth, zone_length))
    dx_CS_MorphoXshore          = np.zeros((Nmonth, zone_length))
    dx_CS_TOTAL                 = np.zeros((Nmonth, zone_length))

    X_CS_MorphoLST              = np.zeros((Nmonth, zone_length))
    X_CS_MorphoTOT              = np.zeros((Nmonth, zone_length))
    X_CS_Hydro                  = np.zeros((Nmonth, zone_length))
    X_CS_MorphoXshore           = np.zeros((Nmonth, zone_length))
    X_CS_TOTAL                  = np.zeros((Nmonth, zone_length))

    Lon                         = np.zeros((Nmonth, zone_length), dtype=np.float64)
    Lat                         = np.zeros((Nmonth, zone_length), dtype=np.float64)
    Lon_m                       = np.zeros((Nmonth, zone_length), dtype=np.float64)
    Lat_m                       = np.zeros((Nmonth, zone_length), dtype=np.float64)
    DLon                        = np.zeros((Nmonth, zone_length))
    DLat                        = np.zeros((Nmonth, zone_length))
    dLon                        = np.zeros((Nmonth, zone_length))
    dLat                        = np.zeros((Nmonth, zone_length))

    # b. PRE-PROCESS OF LAT/LON DATA - AS WE CONVERT THEM IN METERS WE TAKE THE MID VALUES AS A REFERENCE
    lon_mid                      = np.mean(lon[index])
    lat_mid                      = np.mean(lat[index])

    # c. INITIALISATION WHEN T=0
    Lon[0, :]                    = lon[index]
    Lat[0, :]                    = lat[index]

        # AZIMUTH ANGLE
    alpha[0, :-1]                = FUNC.calculate_azimuth(Lat[0, :], Lon[0, :])
    alpha[0, -1]                 = alpha[0, -2]
    
        # NORMAL COASTLINE ANGLE
    normal[0, :]                 = FUNC.calculate_oriented_normal(alpha[0, :], Dir[0, index])

        # LON/LAT CONVERSION FROM ° TO METERS
    Lon_m[0, :], Lat_m[0, :]     = FUNC.lonlat2xy(Lon[0, :], Lat[0, :], lon_mid, lat_mid)

        # DISTANCE TWO CONSECUTIVE POINT CALCULTION IN METERS
    dLon[0, :-1]                 = np.diff(Lon_m[0, :])
    dLat[0, :-1]                 = np.diff(Lat_m[0, :])
    dLon[0, -1]                  = 0
    dLat[0, -1]                  = 0

    Ls[0, :]                     = np.sqrt(dLon[0, :] ** 2 + dLat[0, :] ** 2)


    # d. TIME LOOP
    for t in range(1, Nmonth-1):
        # UPDATE OF LON/LAT TAKING AS A REFERENCE THE LON/LAT COORDINATE AT T-1
        Lon[t, :]                = Lon[t - 1, :]
        Lat[t, :]                = Lat[t - 1, :]
        
        # ANGLE CALCULATION
        alpha[t, :-1]            = FUNC.calculate_azimuth(Lat[t, :], Lon[t, :])
        alpha[t, -1]             = alpha[t, -2]
        
        normal[t, :]             = FUNC.calculate_oriented_normal(alpha[t, :], Dir[0, index])
        incidence_angle[t, :]    = (np.radians(Dir[t, index]) - normal[t, :])


        # LON/LAT CONVERSION IN METERS
        Lon_m[t, :], Lat_m[t, :] = FUNC.lonlat2xy(Lon[t, :], Lat[t, :], lon_mid, lat_mid)
        
        # DISTANCE TWO CONSECUTIVE POINT CALCULTION IN METERS

        dLon[t, :-1]             = np.diff(Lon_m[t, :])
        dLat[t, :-1]             = np.diff(Lat_m[t, :])

        Ls[t, :-1]               = np.sqrt(dLon[t, :-1] ** 2 + dLat[t, :-1] ** 2)

        smooth_Ls                = 1.5 * Ls[t, :]  # m
        
        
        #d.1. FOR EACH COASTPOINT WE COMPUTE THE DELTA COASTLINE VARIABILITY
        for j in range(zone_length - 1):
            idx                  = index[j]                            #we associate the section point index to the global dataset 
            
            
            # TOTAL WATER LEVEL COMPUTATION TAKING INTO ACCOUNT THE LOCAL INCIDENCE ANGLE
            TWL[t,j]             = SLA[t,idx] + DAC[t,idx] + (np.cos(incidence_angle[t,j])*SU[t,idx]) 
            
            dTWL[t, j]           = TWL[t,j] - TWL[t-1,j]
            dx_CS_Hydro[t, j]    = - (TWL[t,j] - TWL[t-1,j]) / (np.tan(beta[t, idx]))  # m

            # MORPHOLOGICAL COMPONENT - LONGSHORE TRANSPORT RATE (KAMPHUIS, 1991)
            incidence_angle_ip1  = incidence_angle[t, j + 1]
            incidence_angle_i    = incidence_angle[t, j]  # radians

            KAMP_mass_i          = 2.33 * (rohs / (rohs - roh)) * (Tp[t, idx] ** 1.5) * (np.tan(beta[t, idx]) ** 0.75) * (d50 ** -0.25) * (Hs[t, idx] ** 2) * abs(np.sin(2 * incidence_angle_i)) ** 0.6 * np.sign(incidence_angle_i)
            KAMP_i               = 86400 * 30 * (KAMP_mass_i / (rohs - roh)) / (1.0 - poro)  # m3/month

            KAMP_mass_ip1        = 2.33 * (rohs / (rohs - roh)) * (Tp[t, idx + 1] ** 1.5) * (np.tan(beta[t, idx + 1]) ** 0.75) * (d50 ** -0.25) * (Hs[t, idx + 1] ** 2) * abs(np.sin(2 * incidence_angle_ip1)) ** 0.6 * np.sign(incidence_angle_ip1)
            KAMP_ip1             = 86400 * 30 * (KAMP_mass_ip1 / (rohs - roh)) / (1.0 - poro)  # m3/month

            DKAMP                = KAMP_ip1 - KAMP_i
            dKAMP[t, j]          = DKAMP

            dx_CS_MorphoTOT[t, j] = ((-1 / DoC[t,idx]) * ((DKAMP + QrivD[t, idx]) / Ls[t, j]) + ((1 / (np.tan(beta[t, idx]))) - (1 / (np.tan(beta[t - 1, idx]))))) * dt
            dx_CS_MorphoLST[t, j] = ((-1 / DoC[t,idx]) * ((DKAMP + QrivD[t, idx]) / Ls[t, j])) * dt
            dx_CS_MorphoXshore[t, j] = ((1 / (np.tan(beta[t,idx]))) - (1 / (np.tan(beta[t - 1, idx])))) * dt
            
            # TOTAL DELTA
            dx_CS_TOTAL[t,j]      = dx_CS_Hydro[t,j] + dx_CS_MorphoTOT[t,j]

        
        # WHEN ALL DELTA POSITION COMPUTED WE ADD THEM FOR EACH TIME STEP TO THE PREVIOUS CROSS-SHORE POSITION (m)
            # COMPONENT
        X_CS_Hydro[t, :]          = X_CS_Hydro[t - 1, :] + dx_CS_Hydro[t, :]
        X_CS_MorphoTOT[t, :]      = X_CS_MorphoTOT[t - 1, :] + dx_CS_MorphoTOT[t, :]
        X_CS_MorphoLST[t, :]      = X_CS_MorphoLST[t - 1, :] + dx_CS_MorphoLST[t, :]
        X_CS_MorphoXshore[t, :]   = X_CS_MorphoXshore[t - 1, :] + dx_CS_MorphoXshore[t, :]

            # TOTAL
        X_CS_TOTAL[t,:]           = X_CS_TOTAL[t-1,:] + dx_CS_TOTAL[t,:]
        
        # PROJECTION FROM CARTESIAN TO GEOGRAPHICAL (meters)
        DLon[t,:]                 = - dx_CS_TOTAL[t,:]  * np.sin(normal[t,:])
        DLat[t,:]                 =   dx_CS_TOTAL[t,:]  * np.cos(normal[t,:])

        DLon[t,:]                 = FUNC.Wfilter(DLon[t,:],Lon_m[t,:],Lat_m[t,:],smooth_Ls)
        DLat[t,:]                 = FUNC.Wfilter(DLat[t,:],Lon_m[t,:],Lat_m[t,:],smooth_Ls)
        
        # UPDATING THE COORDINATES 
        Lon_m[t,:]                = Lon_m[t,:] + DLon[t,:]
        Lat_m[t,:]                = Lat_m[t,:] + DLat[t,:]
        
        # CONVERTING IN °
        Lon[t,:],Lat[t,:]         = FUNC.xy2lonlat(Lon_m[t,:], Lat_m[t,:], lon_mid, lat_mid)

    #e. FOR EACH SECTION WE STOCK THE RESULTS
    r_index.append(index)
    r_dKAMP.append(dKAMP)
    r_dTWL.append(dTWL)
    r_TWL.append(TWL)

    r_Ls.append(Ls)
    r_alpha.append(alpha)
    r_normal.append(normal)
    r_incidence_angle.append(incidence_angle)
    
    r_dx_CS_Hydro.append(dx_CS_Hydro)
    r_dx_CS_MorphoLST.append(dx_CS_MorphoLST)
    r_dx_CS_MorphoTOT.append(dx_CS_MorphoTOT)
    r_dx_CS_MorphoXshore.append(dx_CS_MorphoXshore)
    r_dx_CS_TOTAL.append(dx_CS_TOTAL)
    
    r_X_CS_MorphoLST.append(X_CS_MorphoLST)
    r_X_CS_MorphoTOT.append(X_CS_MorphoTOT)
    r_X_CS_Hydro.append(X_CS_Hydro)
    r_X_CS_MorphoXshore.append(X_CS_MorphoXshore)
    r_X_CS_TOTAL.append(X_CS_TOTAL)

    r_Lon.append(Lon)
    r_Lat.append(Lat)
    r_Lon_m.append(Lon_m)
    r_Lat_m.append(Lat_m)
    r_DLon.append(DLon)
    r_DLat.append(DLat)
    r_dLon.append(dLon)
    r_dLat.append(dLat)


# In[12]:


# CONCATENATION OF THE RESULTS
# ---------------------------------------------------------------------------------------
results_dKAMP              = np.concatenate(r_dKAMP, axis=1)
results_dTWL               = np.concatenate(r_dTWL, axis=1)
results_TWL                = np.concatenate(r_TWL, axis=1)

results_Ls                 = np.concatenate(r_Ls, axis=1)
results_alpha              = np.concatenate(r_alpha, axis=1)

results_normal             = np.concatenate(r_normal, axis=1)
results_incidence_angle    = np.concatenate(r_incidence_angle, axis=1)

results_dx_CS_Hydro        = np.concatenate(r_dx_CS_Hydro, axis=1)
results_dx_CS_MorphoLST    = np.concatenate(r_dx_CS_MorphoLST, axis=1)
results_dx_CS_MorphoTOT    = np.concatenate(r_dx_CS_MorphoTOT, axis=1)
results_dx_CS_MorphoXshore = np.concatenate(r_dx_CS_MorphoXshore, axis=1)
results_dx_CS_TOTAL        = np.concatenate(r_dx_CS_TOTAL, axis=1)

results_X_CS_MorphoLST     = np.concatenate(r_X_CS_MorphoLST, axis=1)
results_X_CS_MorphoTOT     = np.concatenate(r_X_CS_MorphoTOT, axis=1)
results_X_CS_Hydro         = np.concatenate(r_X_CS_Hydro, axis=1)
results_X_CS_MorphoXshore  = np.concatenate(r_X_CS_MorphoXshore, axis=1)
results_X_CS_TOTAL         = np.concatenate(r_X_CS_TOTAL, axis=1)
  
results_Lon                = np.concatenate(r_Lon, axis=1)
results_Lat                = np.concatenate(r_Lat, axis=1)
results_Lon_m              = np.concatenate(r_Lon_m, axis=1)
results_Lat_m              = np.concatenate(r_Lat_m, axis=1)
results_DLon               = np.concatenate(r_DLon, axis=1)
results_DLat               = np.concatenate(r_DLat, axis=1)
results_dLon               = np.concatenate(r_dLon, axis=1)
results_dLat               = np.concatenate(r_dLat, axis=1)
results_index              = np.concatenate(r_index)

# Calibration of our seasonal cycles 
date_list= pd.date_range('2000-1-1','2019-12-31', freq='ME').strftime("%Y-%m-%d")
date = pd.DatetimeIndex(date_list)
num_dates = len(date_list)

VALIDATION_POSITION = {f"Position_{i}": Xshores_val[:, i] for i in range(lon_len)}
VALIDATION = pd.DataFrame(VALIDATION_POSITION, index=pd.to_datetime(date_list))

MODELE_POSITION = {f"Position_{i}": X_CS_TOTAL[:, i] for i in range(lon_len)}
MODELE = pd.DataFrame(MODELE_POSITION, index=pd.to_datetime(date_list))

def compute_seasonal_cycle_np(X, dates):
    """
    Calcule le cycle saisonnier moyen d'une série temporelle.

    Paramètres:
    - X (array): Série temporelle.
    - dates (pd.DatetimeIndex): Index temporel contenant les dates.

    Retourne:
    - seasonal_cycle (array): Cycle saisonnier moyen (taille = 12 mois).
    """
    months = dates.month  # Extraire les mois
    seasonal_cycle = np.array([np.mean(X[months == m]) for m in range(1, 13)])  # Moyenne par mois
    return seasonal_cycle

#  Fonction pour ajuster MODELE avant le calcul du cycle saisonnier
def transform_MODELE(c, X_MODELE):
    """
    Applique une transformation linéaire à X_MODELE.

    Paramètres:
    - c (float): Coefficient multiplicatif.
    - X_MODELE (array): Série temporelle originale.

    Retourne:
    - X_transformed (array): Série ajustée.
    """
    return X_MODELE * c  # Multiplication par c pour ajustement

#  Fonction pour calculer la RMSE après transformation et cycle saisonnier
def compute_rmse_c(c, X_MODELE, X_VALIDATION, dates):
    """
    Transforme X_MODELE, calcule son cycle saisonnier et compare avec X_VALIDATION.

    Paramètres:
    - c (float): Coefficient multiplicatif.
    - X_MODELE (array): Série temporelle originale.
    - X_VALIDATION (array): Cycle saisonnier de validation (taille 12).
    - dates (pd.DatetimeIndex): Index temporel contenant les dates.

    Retourne:
    - RMSE (float): Erreur quadratique moyenne.
    """
    X_transformed = transform_MODELE(c, X_MODELE)  # Appliquer la transformation
    seasonal_cycle_MODELE = compute_seasonal_cycle_np(X_transformed, dates)  # Calcul du cycle saisonnier
    seasonal_cycle_VALIDATION = compute_seasonal_cycle_np(X_VALIDATION, dates)  # Cycle saisonnier de validation
    
    return np.sqrt(np.mean((seasonal_cycle_MODELE - seasonal_cycle_VALIDATION) ** 2))  # Calcul de la RMSE
# Liste des positions à analyser
positions_to_plot = [f"Position_{i}" for i in range(lon_len)]

# Stocker les résultats optimaux
optimal_c_values = {}

#  Boucle sur chaque position pour optimiser c
for pos in positions_to_plot:
    X_MODELE = MODELE[pos].values  # Extraire la série MODELE (array)
    X_VALIDATION = VALIDATION[pos].values  # Extraire la série VALIDATION (array)
    dates = MODELE.index  # Récupérer les dates

    # Recherche du meilleur c qui minimise la RMSE
    result = minimize_scalar(
        compute_rmse_c, 
        bounds=(-50, 50),  # Plage de recherche
        args=(X_MODELE, X_VALIDATION, dates), 
        method="bounded"
    )

    # Extraction du c optimal
    c_optimal = result.x
    rmse_min = result.fun
    optimal_c_values[pos] = c_optimal  # Stocker le meilleur c

    print(f" {pos} - Meilleur c: {c_optimal:.6f}, RMSE minimale: {rmse_min:.10f}")

    #  Visualisation de l'évolution de la RMSE en fonction de c
    c_values = np.linspace(c_optimal - 2, c_optimal + 2, 500)  # Générer plusieurs valeurs de c
    rmse_values = [compute_rmse_c(c, X_MODELE, X_VALIDATION, dates) for c in c_values]  # Calcul de la RMSE

    plt.figure(figsize=(10, 6))
    plt.plot(c_values, rmse_values, label="RMSE vs c", linewidth=2)
    plt.axvline(c_optimal, color='red', linestyle='--', label=f"Optimal c = {c_optimal:.6f}")
    plt.xlabel("Coefficient c")
    plt.ylabel("RMSE")
    plt.title(f"Évolution de la RMSE pour {pos}")
    plt.legend()
    plt.grid()
    plt.show()

    #  Comparaison des cycles saisonniers avant et après ajustement
    X_transformed_optimal = transform_MODELE(c_optimal, X_MODELE)
    seasonal_cycle_optimal = compute_seasonal_cycle_np(X_transformed_optimal, dates)
    seasonal_cycle_original = compute_seasonal_cycle_np(X_MODELE, dates)
    seasonal_cycle_validation = compute_seasonal_cycle_np(X_VALIDATION, dates)

    plt.figure(figsize=(10, 6))
    plt.plot(seasonal_cycle_validation, label="Cycle Validation (Xval)", color="blue", linewidth=2)
    plt.plot(seasonal_cycle_original, label="Cycle MODELE Original", color="gray", linestyle="--", alpha=0.7)
    plt.plot(seasonal_cycle_optimal, label=f"Cycle Ajusté (c={c_optimal:.6f})", color="red", linewidth=2)

    plt.xlabel("Mois")
    plt.ylabel("Valeur")
    plt.title(f"Comparaison des Cycles Saisonniers pour {pos}")
    plt.legend()
    plt.grid()
    plt.show()
    
    #  Affichage final des meilleurs coefficients c
print("\n Meilleurs coefficients c trouvés pour chaque position :")
for pos, c_value in optimal_c_values.items():
    print(f"{pos}: c = {c_value:.6f}")

X_GLOBCOASTS_CALIBRE = c_value * X_CS_TOTAL

# In[13]:


# POST TREATMENT OF THE RESULTS
# ---------------------------------------------------------------------------------------
# a. TEMPORAL SMOOTH OVER 3 MONTHS 
# Define the sigma (standard deviation) for Gaussian smoothing
# Assuming an approximate 30-day window, adjust sigma based on your data frequency

sigma = 1.0  # This will smooth over approximately 3 months, adjust as needed

X_TOTAL_detrend             = FUNC.gaussian_smooth(results_X_CS_TOTAL, sigma)
X_Hydro_detrend             = FUNC.gaussian_smooth(results_X_CS_Hydro, sigma)
X_MorphoTOT_detrend         = FUNC.gaussian_smooth(results_X_CS_MorphoTOT, sigma)
X_MorphoXshore_detrend      = FUNC.gaussian_smooth(results_X_CS_MorphoXshore, sigma)
X_MorphoLST_detrend         = FUNC.gaussian_smooth(results_X_CS_MorphoLST,sigma)

X_val_detrend               = FUNC.gaussian_smooth(Xshores_val, sigma)

dx_Hydro_detrend            = FUNC.gaussian_smooth(results_dx_CS_Hydro, sigma)
dx_MorphoLST_detrend        = FUNC.gaussian_smooth(results_dx_CS_MorphoLST, sigma)
dx_MorphoXshore_detrend     = FUNC.gaussian_smooth(results_dx_CS_MorphoXshore, sigma)
dx_MorphoTOT_detrend        = FUNC.gaussian_smooth(results_dx_CS_MorphoTOT, sigma)
dx_TOTAL_detrend            = FUNC.gaussian_smooth(results_dx_CS_TOTAL,sigma)

SLA_detrend                 = FUNC.gaussian_smooth(SLA[:,results_index],sigma)
DAC_detrend                 = FUNC.gaussian_smooth(DAC[:,results_index],sigma)
SU_detrend                  = FUNC.gaussian_smooth(SU[:,results_index],sigma)
DKAMP_detrend               = FUNC.gaussian_smooth(results_dKAMP,sigma)
QrivD_detrend               = FUNC.gaussian_smooth(QrivD[:,results_index],sigma)


# In[14]:


# b. LINEAR DETREND
X_MorphoLST_detrend_linear    = np.apply_along_axis(detrend_linear,0,X_MorphoLST_detrend)
X_Hydro_detrend_linear        = np.apply_along_axis(detrend_linear,0,X_Hydro_detrend)
X_MorphoXshore_detrend_linear = np.apply_along_axis(detrend_linear,0,X_MorphoXshore_detrend)
X_MorphoTOT_detrend_linear    = np.apply_along_axis(detrend_linear,0,X_MorphoTOT_detrend)
X_TOTAL_detrend_linear        = np.apply_along_axis(detrend_linear,0,X_TOTAL_detrend)
X_val_detrend_global          = np.apply_along_axis(detrend_linear,0,Xshores_val)


dx_Hydro_detrend_linear       = np.apply_along_axis(detrend_linear,0,dx_Hydro_detrend)
dx_MorphoLST_detrend_linear   = np.apply_along_axis(detrend_linear,0,dx_MorphoLST_detrend)
dx_MorphoXshore_detrend_linear= np.apply_along_axis(detrend_linear,0,dx_MorphoXshore_detrend)
dx_MorphoTOT_detrend_linear   = np.apply_along_axis(detrend_linear,0,dx_MorphoTOT_detrend)
dx_TOTAL_detrend_linear       = np.apply_along_axis(detrend_linear,0,dx_TOTAL_detrend)


SLA_detrend_linear            = np.apply_along_axis(detrend_linear,0,SLA_detrend)
DAC_detrend_linear            = np.apply_along_axis(detrend_linear,0,DAC_detrend)
SU_detrend_linear             = np.apply_along_axis(detrend_linear,0,SU_detrend)
DKAMP_detrend_linear          = np.apply_along_axis(detrend_linear,0,DKAMP_detrend)
QrivD_detrend_linear          = np.apply_along_axis(detrend_linear,0,QrivD_detrend)


# In[ ]:





# In[ ]:





# In[ ]:




