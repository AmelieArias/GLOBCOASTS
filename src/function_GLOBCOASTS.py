# -*- coding: utf-8 -*-
"""
Created for GLOBCOASTS Python Script
Purpose: Contains reusable functions for data preprocessing, geospatial transformations, and statistical calculations.
"""

import numpy as np
import pandas as pd
import pyproj
from pyproj import Geod
from math import radians, sin, cos, sqrt, atan2
import matplotlib as plt
from scipy.ndimage import gaussian_filter1d

# ----------------------------------------------
# Function Definitions
# ----------------------------------------------

def shape_waves(time_data, wave_height, wave_period, wave_direction, secondary_wave_direction):
    """
    Purpose: Processes and reshapes wave data into monthly averages.

    Inputs:
        - time_data: Array of time data (numpy array)
        - wave_height: Significant wave height data (numpy array)
        - wave_period: Wave period data (numpy array)
        - wave_direction: Wave direction data (numpy array)
        - secondary_wave_direction: Secondary wave direction data (numpy array)

    Returns:
        - Waves: Dictionary containing reshaped monthly wave data.
    """
    time_data = time_data[5114:13879]
    time_data = time_data.reshape(8765)
    time_data = [pd.to_datetime(i.astype('str'), format="%Y%m%d") for i in time_data]

    wave_height = pd.DataFrame(data=wave_height[5114:13879], index=time_data).resample('M').mean().to_numpy()
    wave_period = pd.DataFrame(data=wave_period[5114:13879], index=time_data).resample('M').mean().to_numpy()
    wave_direction = pd.DataFrame(data=wave_direction[5114:13879], index=time_data).resample('M').mean().to_numpy()
    secondary_wave_direction = pd.DataFrame(data=secondary_wave_direction[5114:13879], index=time_data).resample('M').mean().to_numpy()

    waves = {
        'Hs_monthly': wave_height,
        'Tp_monthly': wave_period,
        'Md_monthly': wave_direction,
        'Mdi_monthly': secondary_wave_direction
    }
    return waves

def calculate_seasonal_climatology(dataframes, date_column='date'):
    """
    Purpose: Computes seasonal climatology from a set of dataframes.

    Inputs:
        - dataframes: Dictionary of pandas DataFrames
        - date_column: Name of the date column in each DataFrame

    Returns:
        - seasonal_climatology: Dictionary containing climatological averages.
    """
    seasonal_climatology = {}
    for name, dataframe in dataframes.items():
        if date_column in dataframe.columns:
            dataframe.set_index(date_column, inplace=True)
        dataframe['Month'] = dataframe.index.month
        seasonal_climatology[name] = dataframe.groupby('Month').mean()
    return seasonal_climatology

def moving_average(data, window_size):
    """
    Purpose: Applies a moving average filter to smooth data.

    Inputs:
        - data: Array of data points (numpy array)
        - window_size: Size of the moving window (integer)

    Returns:
        - smoothed_data: Smoothed data array.
    """
    return np.convolve(data, np.ones(window_size) / window_size, mode='same')

def haversine(coord1, coord2):
    """
    Purpose: Calculates the geodesic distance between two geographic points using the Haversine formula.

    Inputs:
        - coord1: Tuple of (latitude, longitude) for the first point
        - coord2: Tuple of (latitude, longitude) for the second point

    Returns:
        - distance: Distance in kilometers (float).
    """
    lat1, lon1 = radians(coord1[0]), radians(coord1[1])
    lat2, lon2 = radians(coord2[0]), radians(coord2[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return 6371 * c

def nearest_neighbor(coordinates):
    """
    Purpose: Finds the shortest path through a set of geographic coordinates.

    Inputs:
        - coordinates: List of (latitude, longitude) tuples

    Returns:
        - path: List of coordinates representing the shortest path.
    """
    if not coordinates:
        return []
    path = [coordinates.pop(0)]
    while coordinates:
        next_point = min(coordinates, key=lambda x: haversine(path[-1], x))
        path.append(next_point)
        coordinates.remove(next_point)
    return path

def lonlat2xy(lon, lat, lon_0, lat_0):
    """
    Purpose: Converts longitude and latitude to Cartesian coordinates with a specified origin.

    Inputs:
        - lon: Longitude array (numpy array)
        - lat: Latitude array (numpy array)
        - lon_origin: Origin longitude (float)
        - lat_origin: Origin latitude (float)

    Returns:
        - x, y: Cartesian coordinates (numpy arrays).
    """
    geod = Geod(ellps="WGS84")
    lon_0_ar, lat_0_ar = np.full_like(lon, lon_0), np.full_like(lat, lat_0)

    # Calculate y coordinates 
    yd = np.where(lat < lat_0, -1, 1) * geod.inv(lon_0_ar, lat_0_ar, lon_0_ar, lat)[2]

    # Calculate x coordinates 
    xd = np.where(lon < lon_0, -1, 1) * geod.inv(lon_0_ar, lat_0_ar, lon, lat_0_ar)[2]

    return xd, yd

def xy2lonlat(x, y, lon_0, lat_0):
   """
   Purpose: Converts Cartesian coordinates back to longitude and latitude.

   Inputs:
       - x: Cartesian x-coordinates (numpy array)
       - y: Cartesian y-coordinates (numpy array)
       - lon_origin: Origin longitude (float)
       - lat_origin: Origin latitude (float)

   Returns:
       - lon, lat: Longitude and latitude arrays (numpy arrays).
   """
    # Create Geod object for distance calculation
   geod = Geod(ellps="WGS84")

   lon_0_ar, lat_0_ar = np.full_like(x, lon_0), np.full_like(y, lat_0)

    # Calculate longitudes
   lon = geod.fwd(lon_0_ar, lat_0_ar, np.full_like(x, 90), x)[0]

    # Calculate latitudes
   lat = geod.fwd(lon_0_ar, lat_0_ar, np.zeros_like(y), y)[1]

   return lon, lat


def calculate_azimuth(latitudes, longitudes):
    """
    Purpose: Calculates azimuth angles between successive geographic points.

    Inputs:
        - latitudes: Latitude array (numpy array)
        - longitudes: Longitude array (numpy array)

    Returns:
        - azimuth: Azimuth angles in radians (numpy array).
    """
    azimuth = np.zeros(len(longitudes) - 1)
    for i in range(len(longitudes) - 1):
        lat1 = np.radians(latitudes[i])
        lat2 = np.radians(latitudes[i + 1])
        d_lon = np.radians(longitudes[i + 1] - longitudes[i])
        x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(d_lon)
        y = np.sin(d_lon) * np.cos(lat2)
        azimuth[i] = np.arctan2(y, x)
    return azimuth

def calculate_oriented_normal(azimuth, wave_directions):
    """
    Purpose: Computes oriented normals based on azimuth and wave directions.

    Inputs:
        - azimuth: Azimuth angles in radians (numpy array)
        - wave_directions: Wave direction angles in degrees (numpy array)

    Returns:
        - oriented_normals: Oriented normals in radians (numpy array).
    """
    wave_directions_rad = np.radians(wave_directions)
    normal_1 = (azimuth + np.pi / 2) % (2 * np.pi)
    normal_2 = (azimuth - np.pi / 2) % (2 * np.pi)
    oriented_normals = np.zeros_like(azimuth)
    for i in range(len(azimuth)):
        if abs(normal_1[i] - wave_directions_rad[i]) < abs(normal_2[i] - wave_directions_rad[i]):
            oriented_normals[i] = normal_1[i]
        else:
            oriented_normals[i] = normal_2[i]
    return oriented_normals

def calculate_hmax_annual(wave_height_data):
    """
    Purpose: Computes the annual maximum wave height for each position and repeats it for all months of the year.

    Inputs:
        - wave_height_data: Significant wave height data (numpy array) with shape [time, position].

    Returns:
        - df_hmax: DataFrame containing repeated annual maximum wave height for each position.
    """
    n_months_per_year = 12
    n_years = wave_height_data.shape[0] // n_months_per_year

    # Reshape data to separate years
    reshaped_data = wave_height_data[:n_years * n_months_per_year, :].reshape(n_years, n_months_per_year, -1)

    # Calculate annual maximum wave height
    hmax_annual = reshaped_data.max(axis=1)

    # Repeat annual maximum wave height for all months
    hmax_repeated = np.repeat(hmax_annual, n_months_per_year, axis=0)

    # Convert to DataFrame
    positions = [f"Position_{i}" for i in range(wave_height_data.shape[1])]
    time_index = pd.date_range(start="2000-01", periods=wave_height_data.shape[0], freq="M")
    df_hmax = pd.DataFrame(hmax_repeated, columns=positions, index=time_index)
    df_hmax.index.name = 'Time'

    return df_hmax

def Wfilter(data, x, y, distance): 
    filteredData = np.zeros_like(data)                            # np.zeros(data) --> ca me donne un array de 251,288

    for i in range(len(data)):                                   # pour chaque position
        D = np.sqrt((x[i] - x)**2 + (y[i] - y)**2)               #La valeur D(distance) entre les coordonnées LAT/LON de notre données
        
        ii = np.where(D < distance)[0]                           #On va aller chercher les différents indices pour lesquels la distance entre LAT/LON est inf
                                                        # a la distance donnée dans l'appelle de la fonction
        # Calculate mean
        filteredData[i] = np.nanmean(data[ii])                   #On associe a filteredData, la moyenne des datas en ii 
  
        
    return filteredData

def find_index(lon, lat, seuil_distance):
    
    d_lat = [lat[i+1]-lat[i] for i in range(len(lat)-1)]
    d_lat = np.concatenate(([0],d_lat))
    d_lon = [lon[i+1]-lon[i] for i in range(len(lon)-1)]
    d_lon = np.concatenate(([0],d_lon))
    
    start_index = [0]
    end_index   = []

    L = np.sqrt((d_lat)**2+(d_lon)**2)
    
    for i in range(len(L)):
        if L[i] > seuil_distance:
            end_index.append(i)
            start_index.append(i)

    end_index.append(len(lon))

    return start_index, end_index

def distance_between_section(section1, section2):
    """
    Calcule la distance entre la dernière valeur de section1 et la première valeur de section2.
    """
    return np.linalg.norm(section1[-1] - section2[0])
def join_sections(start_index, end_index, lon, lat, distance_threshold):
 # Liste pour stocker les sections
    sections = []

    # Utilisez les index_deb et index_fin pour accéder aux sections et les stocker dans la liste
    for i in range(len(start_index)):
        section_lon = lon[start_index[i]:end_index[i]]
        section_lat = lat[start_index[i]:end_index[i]]
        section = np.column_stack((section_lon, section_lat))
        sections.append(section)

    # Liste pour stocker les sections fusionnées
    join_section = [sections[0]]
    join_index   = [(start_index[0],end_index[0])]

    # Parcourir la liste de sections
    for i in range(1, len(sections)):
        distance = distance_between_section(join_section[-1], sections[i])
        
        # Vérifier si la distance est inférieure ou égale à la valeur seuil (0.5 par défaut)
        if distance <= distance_threshold:
            # Si la distance est inférieure ou égale à la valeur seuil, fusionner les sections
            join_section[-1] = np.concatenate((join_section[-1], sections[i]))
        else:
            # Sinon, ajouter la section à la liste des sections fusionnées
            join_section.append(sections[i])
            join_index.append((start_index[i],end_index[i]))
            
    join_section = [section for section in join_section if len(section) > 1]
# Filtrer les indices fusionnés ayant une longueur supérieure à 2
    join_index = [indices for indices in join_index if indices[1] - indices[0] > 1]
    
    return join_section, join_index

def plot_join_section(join_section):
    # Afficher chaque section fusionnée de manière optimale
    for i, section in enumerate(join_section):
        plt.plot(section[:, 0], section[:, 1], label=f'Join section {i + 1}')

    # Ajouter des détails de tracé (titres, légendes, etc.)
    plt.title('Join section with a distance <= 0.8°')
    plt.xlabel('Longitude [°]')
    plt.ylabel('Latitude [°]')
    plt.legend(loc='center left',bbox_to_anchor=(1,0.5))
    plt.show()
    
def RIVDIS(RiverD,BQART):
    Qmean = np.nanmean(RiverD, axis=0)

    mask = np.logical_or(Qmean == 0, np.isnan(Qmean))
    # Create QrivD based on conditions
    QrivD = np.where(mask, 0, (BQART * RiverD) / Qmean)
    return QrivD

def gaussian_smooth(data, sigma):
    smoothed_data = np.apply_along_axis(lambda m: gaussian_filter1d(m, sigma=sigma), axis=0, arr=data)
    return smoothed_data