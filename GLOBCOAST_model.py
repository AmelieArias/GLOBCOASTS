#!/usr/bin/env python
# coding: utf-8

# Import Modules - Streamlined imports

import os
import sys
import time
import math
import datetime
import configparser
import socket
import getpass
import statistics
import cmcrameri
import numpy as np
import pandas as pd
import scipy as sc
import pylab
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.font_manager as fm
import geopandas as gpd
import xarray as xr
import h5py
import cartopy
import cmocean.cm as cm
from pyproj import Geod
from tqdm import tqdm
from math import *
from datetime import *
from pylab import *
from scipy import interpolate
from scipy.interpolate import interp1d, interp2d, griddata
from scipy.stats import *
from scipy.spatial import distance
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize, minimize_scalar
from scipy.signal import butter, filtfilt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from cartopy import crs as ccrs, feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from cartopy.mpl.gridliner import LongitudeFormatter, LatitudeFormatter
from matplotlib.path import Path
from cartopy.mpl.patch import geos_to_path
from netCDF4 import Dataset
from global_land_mask import globe

# Path definition - Using relative paths for better portability
pathin = ''
pathout = 'RESULTS/'

try:
    PTOOL = pathin + ''
    INPUT = pathin + ''
    sys.path.append(PTOOL)
except ImportError:
    raise ImportError("ERROR path is not defined")

# Import main toolbox - Kept unchanged as requested
try:
    import function_GLOBCOASTS as FUNC
except ImportError:
    raise ImportError("ERROR importing FUNC")

print("# - Toolboxes are OK")

# Constants - Centralized in a dictionary for better management
CONSTANTS = {
    "d50": 10e-3,           # Median grain size [m]
    "poro": 0.4,            # Sand porosity
    "rohs": 2650,           # Sand density [kg/m3]
    "roh": 1000,            # Water density [kg/m3]
    "g": 9.81,              # Gravitational acceleration [m/s2]
    "R": 6371000,           # Earth radius [m]
    "dt": 1                 # Time step (1 month)
}

class GlobCoast:
    """Main coastal dynamics model class."""
    
    def __init__(self, input_path):
        """Initialize model with data paths and constants."""
        self.constants = CONSTANTS
        self.input_path = input_path
        self.data = {}
        self.results = {}
        
    def load_data(self):
        """Load and prepare all input data."""
        
        # Load BQART data
        bqart_brut = np.loadtxt(self.input_path + 'Sorties_Bqart.txt', delimiter=';')
        bqart_m3 = bqart_brut[:, 0][:8841] / 2.650  # Convert to m3
        self.data['BQART'] = bqart_m3 / 12  # m3/yr --> m3/month
        
        # Load SEADATA using xarray
        seadata = xr.open_dataset(self.input_path + 'SEADATA_14140pts_1993_2019-analysed.nc', engine='netcdf4')
        
        # Store whole dataset for reference
        self.data['SEADATA_FULL'] = seadata
        
        # Extract coordinates
        self.data['lon'] = seadata['lon'].values[:8841]
        self.data['lat'] = seadata['lat'].values[:8841]
        
        # Extract variables with time slice [84:] and location slice [:8841]
        self.data['Hs'] = seadata['Hs_mounthly'].values[84:, :8841]
        self.data['Tp'] = seadata['Tp_mounthly'].values[84:, :8841]
        self.data['Dir'] = seadata['dir_mounthly'].values[84:, :8841]
        self.data['SLA'] = seadata['sla_detrend'].values[84:, :8841]
        self.data['DAC'] = seadata['dac_detrend'].values[84:, :8841]
        self.data['RiverD'] = seadata['rivdis_mounthly'].values[84:, :8841]
        
        # Load tidal data
        tide = sc.io.loadmat(self.input_path + 'Tide_Glob.mat')
        self.data['Tide_range'] = tide['Tide_max'][0][:8841]
        
        # Clean tide range NaN values
        self._clean_tide_data()
        
        # Load dynamic depth of closure
        dc = xr.open_dataset(self.input_path + 'Dc.nc', engine='netcdf4')
        self.data['DoC'] = dc['Dc'].values
        
        # Load validation data
        validation = sc.io.loadmat(self.input_path + 'Shorelines_global_20231101_shift.mat')
        self.data['Xshores_val'] = validation['X_safe'][:, 108:-12]
        self.data['latX'] = validation['latX'][:]
        self.data['lonX'] = validation['lonX'][:]
        
        # Set transposed validation data
        self.data['Xshores_VALIDATION'] = np.transpose((-1 * self.data['Xshores_val']))
        
        # Initialize time dimension
        self.n_months = len(self.data['SLA'])
        
        print("# - All files are uploaded...")
    
    def _clean_tide_data(self):
        """Clean NaN values in tide range data."""
        for i in range(1, len(self.data['Tide_range'])):
            if np.isnan(self.data['Tide_range'][i]):
                self.data['Tide_range'][i] = self.data['Tide_range'][i-1]
        print("# - Tide range is cleaned")
    
    def preprocess_inputs(self):
        """Process input data for model computations."""
        # Clean liquid river discharge dataset
        for line in self.data['RiverD']:
            line[np.isnan(line)] = 0
            
        # Calculate solid river discharge variability
        self.data['QrivD'] = FUNC.RIVDIS(self.data['RiverD'], self.data['BQART'])
        for line in self.data['QrivD']:
            if np.any(np.isnan(line)):
                print(line)
        print("# - QrivD is ok")
        
        # Calculate total water level
        g = self.constants['g']
        
        # Wave length calculation
        L_SU = (g / (2 * np.pi)) * self.data['Tp']**2
        self.data['L_SU'] = L_SU
        
        # Beach slope calculation using Sunamura 84 modified by tide
        d50 = self.constants['d50']
        beta = 0.12 * ((np.sqrt(2 * np.pi * d50 * L_SU)) / 
                       (self.data['Hs'] * (1 + (self.data['Tide_range'] / self.data['Hs']))))**(1/2)
        self.data['beta'] = beta
        
        for i in range(len(beta)):
            for j in range(len(beta[0])):
                if np.isnan(beta[i, j]):
                    print(f"NaN in beta at {i}, {j}")
        print("# - Foreshore slopes is ok")
        
        # Wave setup calculation (Stockdon, 2006)
        SU = 0.35 * beta * np.sqrt(self.data['Hs'] * L_SU)
        self.data['SU'] = SU
        print("# - Set up is ok")
        
        # Total water level calculation
        TWL = self.data['SLA'] + self.data['DAC'] + SU
        self.data['TWL'] = TWL
        print("# - Total water level is ok")
    
    def prepare_coastline_sections(self):
        """Identify and prepare coastline sections for processing."""
        # Find sections based on position spacing
        start_index, end_index = FUNC.find_index(
            self.data['lon'], self.data['lat'], 0.55)
        
        # Join sections based on lat/lon
        join_section, join_index = FUNC.join_sections(
            start_index, end_index, self.data['lon'], self.data['lat'], 0.55)
        
        # Filter sections to include only those with more than 2 positions
        self.join_section_filtered = [(i, j) for i, j in join_index if np.abs(i - j) > 2]
    
    def initialize_results_containers(self):
        """Initialize containers for storing processing results."""
        # Lists to store section results
        self.section_results = {
            'index': [],
            'dKAMP': [],
            'dTWL': [],
            'TWL': [],
            'Ls': [],
            'alpha': [],
            'normal': [],
            'incidence_angle': [],
            'dx_CS_Hydro': [],
            'dx_CS_MorphoLST': [],
            'dx_CS_MorphoTOT': [],
            'dx_CS_MorphoXshore': [],
            'dx_CS_TOTAL': [],
            'X_CS_MorphoLST': [],
            'X_CS_MorphoTOT': [],
            'X_CS_Hydro': [],
            'X_CS_MorphoXshore': [],
            'X_CS_TOTAL': [],
            'Lon': [],
            'Lat': [],
            'Lon_m': [],
            'Lat_m': [],
            'DLon': [],
            'DLat': [],
            'dLon': [],
            'dLat': []
        }
    
    def process_coastline_section(self, section, section_index):
        """Process a single coastline section."""
        start_idx, end_idx = section
        index = np.arange(start_idx, end_idx)
        zone_length = len(index)
        
        # Initialize arrays for this section
        section_arrays = self._initialize_section_arrays(zone_length, section_index)

        
        # Get reference coordinates
        lon_mid = np.mean(self.data['lon'][index])
        lat_mid = np.mean(self.data['lat'][index])
        
        # Initialize at t=0
        section_arrays['Lon'][0, :] = self.data['lon'][index]
        section_arrays['Lat'][0, :] = self.data['lat'][index]
        
        # Calculate initial azimuth angle
        section_arrays['alpha'][0, :-1] = FUNC.calculate_azimuth(
            section_arrays['Lat'][0, :], section_arrays['Lon'][0, :])
        section_arrays['alpha'][0, -1] = section_arrays['alpha'][0, -2]
        
        # Calculate initial normal angle
        section_arrays['normal'][0, :] = FUNC.calculate_oriented_normal(
            section_arrays['alpha'][0, :], self.data['Dir'][0, index])
        
        # Convert lon/lat to meters
        section_arrays['Lon_m'][0, :], section_arrays['Lat_m'][0, :] = FUNC.lonlat2xy(
            section_arrays['Lon'][0, :], section_arrays['Lat'][0, :], lon_mid, lat_mid)
        
        # Calculate distance between consecutive points
        section_arrays['dLon'][0, :-1] = np.diff(section_arrays['Lon_m'][0, :])
        section_arrays['dLat'][0, :-1] = np.diff(section_arrays['Lat_m'][0, :])
        section_arrays['dLon'][0, -1] = 0
        section_arrays['dLat'][0, -1] = 0
        
        section_arrays['Ls'][0, :] = np.sqrt(
            section_arrays['dLon'][0, :]**2 + section_arrays['dLat'][0, :]**2)
        
        # Process time steps
        for t in range(1, self.n_months-1):
            self._process_time_step(t, section_arrays, index, lon_mid, lat_mid, zone_length)
        
        # Store results for this section
        for key in self.section_results:
            self.section_results[key].append(section_arrays[key])
        
        return section_arrays
    
    def _initialize_section_arrays(self, zone_length, index):
        """Initialize arrays for a coastline section."""
        arrays = {
            'dKAMP': np.zeros((self.n_months, zone_length)),
            'dTWL': np.zeros((self.n_months, zone_length)),
            'TWL': np.zeros((self.n_months, zone_length)),
            'Ls': np.zeros((self.n_months, zone_length)),
            'alpha': np.zeros((self.n_months, zone_length)),
            'normal': np.zeros((self.n_months, zone_length)),
            'incidence_angle': np.zeros((self.n_months, zone_length)),
            'dx_CS_Hydro': np.zeros((self.n_months, zone_length)),
            'dx_CS_MorphoLST': np.zeros((self.n_months, zone_length)),
            'dx_CS_MorphoTOT': np.zeros((self.n_months, zone_length)),
            'dx_CS_MorphoXshore': np.zeros((self.n_months, zone_length)),
            'dx_CS_TOTAL': np.zeros((self.n_months, zone_length)),
            'X_CS_MorphoLST': np.zeros((self.n_months, zone_length)),
            'X_CS_MorphoTOT': np.zeros((self.n_months, zone_length)),
            'X_CS_Hydro': np.zeros((self.n_months, zone_length)),
            'X_CS_MorphoXshore': np.zeros((self.n_months, zone_length)),
            'X_CS_TOTAL': np.zeros((self.n_months, zone_length)),
            'Lon': np.zeros((self.n_months, zone_length), dtype=np.float64),
            'Lat': np.zeros((self.n_months, zone_length), dtype=np.float64),
            'Lon_m': np.zeros((self.n_months, zone_length), dtype=np.float64),
            'Lat_m': np.zeros((self.n_months, zone_length), dtype=np.float64),
            'DLon': np.zeros((self.n_months, zone_length)),
            'DLat': np.zeros((self.n_months, zone_length)),
            'dLon': np.zeros((self.n_months, zone_length)),
            'dLat': np.zeros((self.n_months, zone_length)),
            'index': [index]
        }
        return arrays
    
    def _process_time_step(self, t, arrays, index, lon_mid, lat_mid, zone_length):
        """Process a single time step for a coastline section."""
        # Update lon/lat from previous time step
        arrays['Lon'][t, :] = arrays['Lon'][t-1, :]
        arrays['Lat'][t, :] = arrays['Lat'][t-1, :]
        
        # Calculate angles
        arrays['alpha'][t, :-1] = FUNC.calculate_azimuth(arrays['Lat'][t, :], arrays['Lon'][t, :])
        arrays['alpha'][t, -1] = arrays['alpha'][t, -2]
        
        arrays['normal'][t, :] = FUNC.calculate_oriented_normal(
            arrays['alpha'][t, :], self.data['Dir'][0, index])
        arrays['incidence_angle'][t, :] = (np.radians(self.data['Dir'][t, index]) - arrays['normal'][t, :])
        
        # Convert to meters
        arrays['Lon_m'][t, :], arrays['Lat_m'][t, :] = FUNC.lonlat2xy(
            arrays['Lon'][t, :], arrays['Lat'][t, :], lon_mid, lat_mid)
        
        # Calculate distances
        arrays['dLon'][t, :-1] = np.diff(arrays['Lon_m'][t, :])
        arrays['dLat'][t, :-1] = np.diff(arrays['Lat_m'][t, :])
        
        arrays['Ls'][t, :-1] = np.sqrt(arrays['dLon'][t, :-1]**2 + arrays['dLat'][t, :-1]**2)
        
        smooth_Ls = 1.5 * arrays['Ls'][t, :]  # m
        
        # Process each coastal point
        for j in range(zone_length - 1):
            idx = index[j]  # Global dataset index
            
            # Calculate TWL with local incidence angle
            arrays['TWL'][t, j] = (self.data['SLA'][t, idx] + self.data['DAC'][t, idx] + 
                                 (np.cos(arrays['incidence_angle'][t, j]) * self.data['SU'][t, idx]))
            
            # Calculate dTWL and hydrodynamic component
            arrays['dTWL'][t, j] = arrays['TWL'][t, j] - arrays['TWL'][t-1, j]
            arrays['dx_CS_Hydro'][t, j] = -(arrays['dTWL'][t, j]) / (np.tan(self.data['beta'][t, idx]))
            
            # Calculate morphological component - Longshore transport
            self._calculate_longshore_transport(t, j, idx, arrays, index)
        
        # Update cross-shore positions
        self._update_cross_shore_positions(t, arrays)
        
        # Calculate projection from cartesian to geographical
        arrays['DLon'][t, :] = -arrays['dx_CS_TOTAL'][t, :] * np.sin(arrays['normal'][t, :])
        arrays['DLat'][t, :] = arrays['dx_CS_TOTAL'][t, :] * np.cos(arrays['normal'][t, :])
        
        # Apply spatial filtering
        arrays['DLon'][t, :] = FUNC.Wfilter(arrays['DLon'][t, :], arrays['Lon_m'][t, :], 
                                          arrays['Lat_m'][t, :], smooth_Ls)
        arrays['DLat'][t, :] = FUNC.Wfilter(arrays['DLat'][t, :], arrays['Lon_m'][t, :], 
                                          arrays['Lat_m'][t, :], smooth_Ls)
        
        # Update coordinates
        arrays['Lon_m'][t, :] = arrays['Lon_m'][t, :] + arrays['DLon'][t, :]
        arrays['Lat_m'][t, :] = arrays['Lat_m'][t, :] + arrays['DLat'][t, :]
        
        # Convert back to degrees
        arrays['Lon'][t, :], arrays['Lat'][t, :] = FUNC.xy2lonlat(
            arrays['Lon_m'][t, :], arrays['Lat_m'][t, :], lon_mid, lat_mid)
    
    def _calculate_longshore_transport(self, t, j, idx, arrays, index):
        """Calculate the longshore sediment transport at a point."""
        # Constants
        rohs = self.constants['rohs']
        roh = self.constants['roh']
        d50 = self.constants['d50']
        poro = self.constants['poro']
        dt = self.constants['dt']
        
        # Incidence angles
        incidence_angle_ip1 = arrays['incidence_angle'][t, j + 1]
        incidence_angle_i = arrays['incidence_angle'][t, j]  # radians
        
        # Calculate KAMP for current point
        KAMP_mass_i = (2.33 * (rohs / (rohs - roh)) * 
                      (self.data['Tp'][t, idx] ** 1.5) * 
                      (np.tan(self.data['beta'][t, idx]) ** 0.75) * 
                      (d50 ** -0.25) * 
                      (self.data['Hs'][t, idx] ** 2) * 
                      np.abs(np.sin(2 * incidence_angle_i)) ** 0.6 * 
                      np.sign(incidence_angle_i))
        
        KAMP_i = 86400 * 30 * (KAMP_mass_i / (rohs - roh)) / (1.0 - poro)  # m3/month
        
        # Calculate KAMP for next point
        KAMP_mass_ip1 = (2.33 * (rohs / (rohs - roh)) * 
                        (self.data['Tp'][t, idx + 1] ** 1.5) * 
                        (np.tan(self.data['beta'][t, idx + 1]) ** 0.75) * 
                        (d50 ** -0.25) * 
                        (self.data['Hs'][t, idx + 1] ** 2) * 
                        np.abs(np.sin(2 * incidence_angle_ip1)) ** 0.6 * 
                        np.sign(incidence_angle_ip1))
        
        KAMP_ip1 = 86400 * 30 * (KAMP_mass_ip1 / (rohs - roh)) / (1.0 - poro)  # m3/month
        
        # Calculate difference
        DKAMP = KAMP_ip1 - KAMP_i
        arrays['dKAMP'][t, j] = DKAMP
        
        # Calculate morphological components
        arrays['dx_CS_MorphoTOT'][t, j] = ((-1 / self.data['DoC'][t, idx]) * 
                                         ((DKAMP + self.data['QrivD'][t, idx]) / arrays['Ls'][t, j]) - 
                                         ((1 / (np.tan(self.data['beta'][t, idx]))) - 
                                          (1 / (np.tan(self.data['beta'][t-1, idx]))))) * dt
        
        arrays['dx_CS_MorphoLST'][t, j] = ((-1 / self.data['DoC'][t, idx]) * 
                                         ((DKAMP + self.data['QrivD'][t, idx]) / arrays['Ls'][t, j])) * dt
        
        arrays['dx_CS_MorphoXshore'][t, j] = -((1 / (np.tan(self.data['beta'][t, idx]))) - 
                                             (1 / (np.tan(self.data['beta'][t-1, idx])))) * dt
        
        # Calculate total delta
        arrays['dx_CS_TOTAL'][t, j] = arrays['dx_CS_Hydro'][t, j] + arrays['dx_CS_MorphoTOT'][t, j]
    
    def _update_cross_shore_positions(self, t, arrays):
        """Update cross-shore positions for each component."""
        # Component updates
        arrays['X_CS_Hydro'][t, :] = arrays['X_CS_Hydro'][t-1, :] + arrays['dx_CS_Hydro'][t, :]
        arrays['X_CS_MorphoTOT'][t, :] = arrays['X_CS_MorphoTOT'][t-1, :] + arrays['dx_CS_MorphoTOT'][t, :]
        arrays['X_CS_MorphoLST'][t, :] = arrays['X_CS_MorphoLST'][t-1, :] + arrays['dx_CS_MorphoLST'][t, :]
        arrays['X_CS_MorphoXshore'][t, :] = arrays['X_CS_MorphoXshore'][t-1, :] + arrays['dx_CS_MorphoXshore'][t, :]
        
        # Total update
        arrays['X_CS_TOTAL'][t, :] = arrays['X_CS_TOTAL'][t-1, :] + arrays['dx_CS_TOTAL'][t, :]
    
    def run_model(self):
        """Run the coastal model for all sections."""
        print("# - Running coastal model...")
        # Process each section
        for i, section in tqdm(enumerate(self.join_section_filtered)):
            self.process_coastline_section(section, i)
        self.concatenate_results()
        self.calibrate_model()
        self.post_process_results()
        print("# - Model run complete.")
    
    def concatenate_results(self):
        """Concatenate results from all sections."""
        # Concatenate along axis 1 (spatial dimension)
        self.results = {key: np.concatenate(self.section_results[key], axis=1) 
                       for key in self.section_results}
        print(self.results)
        # Extract the index for reference
        self.results_index = np.concatenate(self.section_results['index'])
    
    def calibrate_model(self):
        """Calibrate the model against validation data."""
        print("# - Calibrating model against validation data...")
        
        # Format data for calibration
        daily_dates = pd.date_range('2000-01-01', periods=8841, freq='D')
        df_daily = pd.DataFrame(self.data['Xshores_val'], index=daily_dates)
        df_monthly = df_daily.resample('ME').mean()
        
        validation = pd.DataFrame(df_monthly.values, 
                                 index=pd.date_range('2000-01-01', '2024-03-31', freq='ME'))
        model = pd.DataFrame(self.results['X_CS_TOTAL'], 
                            index=pd.date_range('2000-01-01', '2019-12-31', freq='ME'))
        
        print(validation, model)
        
        # Compute optimal calibration coefficient
        lon_len = len(self.data['lon'])
        optimal_c_values = {}
        
        for pos_idx in range(lon_len):
            pos = f"Position_{pos_idx}"
            X_MODEL = model[pos].values if pos in model else np.zeros(len(model))
            X_VALIDATION = validation[pos].values if pos in validation else np.zeros(len(validation))
            dates = model.index
            
            # Find optimal coefficient
            result = minimize_scalar(
                self._compute_rmse_c,
                bounds=(-50, 50),
                args=(X_MODEL, X_VALIDATION, dates),
                method="bounded"
            )
            
            c_optimal = result.x
            rmse_min = result.fun
            optimal_c_values[pos] = c_optimal
            
            print(f"{pos} - Best c: {c_optimal:.6f}, Minimum RMSE: {rmse_min:.10f}")
        
        # Calculate calibrated model
        c_value = np.mean(list(optimal_c_values.values()))
        self.results['X_GLOBCOASTS_CALIBRE'] = c_value * self.results['X_CS_TOTAL']
    
    def _compute_seasonal_cycle_np(self, X, dates):
        """Compute mean seasonal cycle of a time series."""
        months = dates.month
        seasonal_cycle = np.array([np.mean(X[months == m]) for m in range(1, 13)])
        return seasonal_cycle
    
    def _transform_MODEL(self, c, X_MODEL):
        """Apply linear transformation to model data."""
        return X_MODEL * c
    
    def _compute_rmse_c(self, c, X_MODEL, X_VALIDATION, dates):
        """Compute RMSE between transformed model and validation seasonal cycles."""
        X_transformed = self._transform_MODEL(c, X_MODEL)
        seasonal_cycle_MODEL = self._compute_seasonal_cycle_np(X_transformed, dates)
        seasonal_cycle_VALIDATION = self._compute_seasonal_cycle_np(X_VALIDATION, dates)
        
        return np.sqrt(np.mean((seasonal_cycle_MODEL - seasonal_cycle_VALIDATION) ** 2))
    
    def post_process_results(self):
        """Apply post-processing to model results."""
        print("# - Post-processing results...")
        
        # Define Gaussian smoothing sigma
        sigma = 1.0  # Smooths over approximately 3 months
        
        # Apply Gaussian smoothing
        smooth_keys = [
            'X_CS_TOTAL', 'X_CS_Hydro', 'X_CS_MorphoTOT', 
            'X_CS_MorphoXshore', 'X_CS_MorphoLST',
            'dx_CS_Hydro', 'dx_CS_MorphoLST', 'dx_CS_MorphoXshore',
            'dx_CS_MorphoTOT', 'dx_CS_TOTAL'
        ]
        
        for key in smooth_keys:
            self.results[f"{key}_detrend"] = FUNC.gaussian_smooth(self.results[key], sigma)
        
        # Smooth validation data
        self.results['X_val_detrend'] = FUNC.gaussian_smooth(self.data['Xshores_val'], sigma)
        
        # Smooth input variables
        self.results['SLA_detrend'] = FUNC.gaussian_smooth(self.data['SLA'][:, self.results_index], sigma)
        self.results['DAC_detrend'] = FUNC.gaussian_smooth(self.data['DAC'][:, self.results_index], sigma)
        self.results['SU_detrend'] = FUNC.gaussian_smooth(self.data['SU'][:, self.results_index], sigma)
        self.results['DKAMP_detrend'] = FUNC.gaussian_smooth(self.results['dKAMP'], sigma)
        self.results['QrivD_detrend'] = FUNC.gaussian_smooth(self.data['QrivD'][:, self.results_index], sigma)
        
        print("# - Post-processing complete.")

# Main execution
if __name__ == "__main__":
    # Instantiate the model
    model = GlobCoast(pathin)
    model.load_data()
    model.preprocess_inputs()
    model.prepare_coastline_sections() 
    model.initialize_results_containers() 
    model.run_model()
    model.post_process_results()
