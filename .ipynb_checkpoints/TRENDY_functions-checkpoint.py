import os
import glob
import math
import cftime
import dask
import functools
import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


def get_active_pfts(hist_dir, stream='h1'):
    
    pft_files = sorted(glob.glob(f'{hist_dir}/*.{stream}*.nc'))
    
    ds = xr.open_dataset(pft_files[0])
    active_pfts = [int(pft) for pft in np.unique(ds.pfts1d_itype_veg)]
    
    return active_pfts

def calculate_global_sum(da: xr.DataArray, land_area: xr.DataArray=None) -> xr.DataArray:
    """Calculates global sum for a variable, multiplies by land area if supplied

    Args:
        da (xr.DataArray): input DataArray
        land_area (xr.DataArray): land area DataArray, defaults to None

    Returns:
        xr.DataArray: output DataArray
    """
    
    if land_area is not None:
        global_sum = (da*land_area).sum(dim=['lat', 'lon'])
    else:
        global_sum = da.sum(dim=['lat', 'lon'])
    
    return global_sum

def get_h1_files(hist_dir: str, exp: str) -> list[str]:
    """Returns a list of h1 files from an input history directory and TRENDY experiment

    Args:
        hist_dir (str): path to history diretory
        exp (str): TRENDY experiment [S0, S1, ...]

    Returns:
        list[str]: sorted list of files
    """
    return sorted(glob.glob(os.path.join(hist_dir, f'TRENDY2024_f09_clm60_{exp}.clm2.h1.*.nc')))

def read_landcoverfrac_ds(hist_dir: str, exp: str, var_in: str) -> xr.Dataset:
    """Reads in the data needed to convert land cover frac to by pft

    Args:
        hist_dir (str): path to history directory
        exp (str): TRENDY experiment name [S0, S1,...]
        var_in (str): variable to convert

    Returns:
        xr.Dataset: output dataset
    """

    # find the h1 archive files
    h1_files = get_h1_files(hist_dir, exp)

    # open the dataset
    ds = xr.open_mfdataset(h1_files, combine='nested', concat_dim='time', parallel=True, autoclose=True, 
                           preprocess=functools.partial(preprocess, data_vars=[var_in, 'time']),
                           chunks={'time': 10})
    year_zero = ds['time.year'][0].values
    ds['time'] = xr.cftime_range(str(year_zero), periods=len(ds.time), freq='MS', calendar='noleap')

    # add in these variables for h1 streams
    save_vars = ['lat', 'lon', 'pfts1d_itype_veg', 'pfts1d_ixy', 'pfts1d_jxy', 'landfrac']
    tmp = xr.open_dataset(h1_files[0])
    for var in save_vars:
        ds[var] = tmp[var]

    return ds

def regrid_landcoverfrac(ds: xr.Dataset, var_out: str, var: str, data_dict: dict, 
                         pft: int, pft_names: xr.DataArray) -> xr.Dataset:
    """Regrids and returns landcover fraction for an input pft

    Args:
        ds (xr.Dataset): input dataset
        var_out (str): output variable name
        var (str): CLM variable name for PFT land cover fraction
        data_dict (dict): dictionary with information about this variable
        pft (int): PFT index
        pft_names (xr.DataArray): array with PFT names

    Returns:
        xr.Dataset: output dataset
    """
    
    ds_regridded = regrid_h1_ds(ds, pft, var, pft_names)
    ds_annual = calculate_annual_mean(ds_regridded.pfts1d_wtgcell)

    data_out = ds_annual.to_dataset(name=var_out)

    data_out = data_out.rename({'year': 'time'})
    data_out.time.attrs = {'long_name': 'years since 1700'}

    # set attributes
    data_out[var_out].attrs = {
        'long_name': data_dict['long_name'],
        'units': data_dict['output_units'],
        '_FillValue': data_dict['fill_val'],
        'CLM-TRENDY_unit_conversion_factor': data_dict['conversion_factor'],
        'CLM_orig_var_name': var
    }
    data_out[var_out].attrs[f"CLM_orig_attr_{var}_long_name"] = ds.pfts1d_wtgcell.attrs['long_name']
    
    data_out = data_out.transpose('time', 'PFT', 'lat', 'lon')
    data_out['pftname'] = ds_regridded.pftname
    data_out['pftname'].attrs = {'long_name': 'pft name'}
    data_out['PFT'].attrs = {'long_name': 'pft index'}
    data_out = data_out.fillna(data_dict['fill_val'])

    return data_out
            
def get_land_area(hist_dir: str) -> xr.DataArray:
    """Grabs land area and land fraction for a CLM case

    Args:
        hist_dir (str): path to history directory

    Returns:
        xr.DataArray: DataArray of grid area [m2] and grid fraction
    """

    # open a file to get land area
    hist_file = sorted(glob.glob(f'{hist_dir}/*h0*'))[0]
    ds = xr.open_dataset(hist_file)

    # get gridcell area and land area
    grid_area = 1e6*ds.area
    grid_area.attrs = {'long_name': 'grid cell areas', 'units': 'm2'}

    land_area = xr.Dataset()
    land_area['area'] = grid_area
    land_area['landfrac'] = ds.landfrac
    
    return land_area

def round_up(n: float, decimals :int=0) -> int:
    """Helper function to round up a number to a specified number of integers

    Args:
        n (float): input number
        decimals (int, optional): Number to round to. Defaults to 0.

    Returns:
        int: rounded number
    """
    
    multiplier = 10**decimals
    return math.ceil(n*multiplier)/multiplier

def truncate(n: float, decimals :int=0) -> int:
    """Helper function to round down a number to a specified number of integers

    Args:
        n (float): input number
        decimals (int, optional): Number to round to. Defaults to 0.

    Returns:
        int: rounded number
    """
    
    multiplier = 10**decimals
    return int(n*multiplier)/multiplier

def make_NBP_df(top_dir: str, exp: str):
    """Creates and writes a NBP dataset for a TRENDY experiment

    Args:
        tseries_dir (str): path to single-variable time-series folder
        hist_dir (str): path to history folder
        exp (str): TRENDY experiment
        out_dir (str): output directory to write to
    """
    
    # read in NBP
    nbp_file = os.path.join(top_dir, f"CLM6.0_{exp}_nbp.nc")
    nbp_ds = xr.open_dataset(nbp_file)

    # grab land area
    grid_area = xr.open_dataset(os.path.join(top_dir, f"CLM6.0_{exp}_gridArea.nc"))
    ocean_frac = xr.open_dataset(os.path.join(top_dir, f"CLM6.0_{exp}_oceanCoverFrac.nc"))
    land_frac = 1.0 - ocean_frac.oceanCoverFrac
    land_area_global = grid_area.gridArea*land_frac
    
    # create domains
    lats = land_area_global.lat
    domains = {'Global': land_area_global,
              'North': land_area_global.where(lats > 30),
              'Tropics': land_area_global.where(abs(lats) < 30),
              'South': land_area_global.where(lats < -30)}
    
    conversion_factor = 1e-12*24*60*60*365  # PgC/yr
    xs = {region: conversion_factor*calculate_global_sum(calculate_annual_mean(nbp_ds.nbp),
                                                         domains[region]) for region in domains}
    df = pd.DataFrame({region: xs[region] for region in xs},
                  index=xs['Global'].year.values)
    df.index.name = 'Year'

    file_out = os.path.join(top_dir, f'CLM6.0_{exp}_zonalNBP.txt')
    df.to_csv(file_out)
    
def get_all_nbp_df(out_dir: str, exps: list[str]) -> pd.DataFrame:
    """Reads in all the TRENDY NBP dataframes into one dataframe

    Args:
        out_dir (str): path to directory with single-experiment dataframes
        exps (list[str]): list of experiments to include

    Returns:
        pd.DataFrame: output DataFrame
    """
    
    data_frames = []
    for exp in exps:
        file_in = os.path.join(f"{out_dir}/{exp}", f'CLM6.0_{exp}_zonalNBP.txt')
        df = pd.read_csv(file_in)
        df['experiment'] = exp
        data_frames.append(df)
      
    nbp = pd.concat(data_frames)
    nbp = pd.melt(nbp, id_vars=['Year', 'experiment'],
                value_vars=['Global', 'North', 'Tropics', 'South'],
                value_name='NBP', var_name='region')
    
    return nbp

def plot_running_means(df: pd.DataFrame):
    """Plots global running means of NBP for each TRENDY experiments

    Args:
        df (pd.DataFrame): input data frame of global NBP
    """
  
    minval = df.NBP.min()
    if minval < 0:
        minvar = round_up(np.abs(minval))*-1.0
    else:
        minvar = truncate(minval)
    
    maxvar = round_up(df.NBP.max())
    
    tableau = [(31, 119, 180), (255, 127, 14), (44, 160, 44),  (214, 39, 40)]
    for i in range(len(tableau)):
        r, g, b = tableau[i]
        tableau[i] = (r/255., g/255., b/255.)
  
    plt.figure(figsize=(7, 5))
    ax = plt.subplot(111)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    plt.xlim([1720, 2023])
    plt.ylim(minvar, maxvar)
    
    inc = (int(maxvar) - minvar)/20
    for i in range(0, 20):
        plt.plot(range(math.floor(1720), math.ceil(2023)),
                  [minvar + i*inc] * len(range(math.floor(1720), math.ceil(2023))),
                  "--", lw=0.5, color="black", alpha=0.3)
    plt.tick_params(bottom=False, top=False, left=False, right=False)
    
    for i, exp in enumerate(np.unique(df.experiment)):
        this_df = df[df['experiment'] == exp]
        rolling_mean = this_df.NBP.rolling(20).mean()
        plt.plot(this_df.Year, rolling_mean, label=exp, color=tableau[i])
    
    plt.axhline(y=0.1, color='gray', linestyle=':')
    plt.axhline(y=-0.1, color='gray', linestyle=':')
    plt.xlabel('Year')
    plt.ylabel('NBP 20-yr running mean (PgC yr$^{-1}$)')
    plt.legend(loc='upper left')

def plot_cumulative_nbp(df: pd.DataFrame):
    """Plots cumulative NBP for different regions and TRENDY experiments

    Args:
        df (pd.DataFrame): input data frame of NBP
    """
    
    for i, region in enumerate(np.unique(df.region)):
        plt.subplot(2, 2, i+1)
        sub_df = df[df.region == region]
        for j, exp in enumerate(np.unique(sub_df.experiment)):
            this_df = sub_df[sub_df['experiment'] == exp]
            plt.plot(this_df.Year, this_df.NBP.cumsum(), label=exp)
        plt.xlim([1700, 2023])
        if i == 0:
            plt.legend()
        if i % 2 == 0:
            plt.ylabel('Cumulative NBP (PgC)')
        plt.title(region)
        if i < 2:
            plt.xticks(1700+100*np.arange(4), [])
        else:
            plt.xlabel('Year')
            
def get_nbp_drift(df: pd.DataFrame) -> float:
    """Returns the slopex100 of the NBPxyear relationship (i.e. drift)

    Args:
        df (pd.DataFrame): input global NBP dataframe

    Returns:
        float: slopex100 (i.e. drift, according to TRENDY)
    """
    
    x = df[['Year']]
    y = df['NBP']
    
    model = LinearRegression()
    model.fit(x, y)
    
    slp = model.coef_[0]
    
    return slp*100.0

def check_TRENDY_criteria(global_df, drift_tol=0.05, offset_tol=0.10):
    
    s0_df = global_df[global_df.experiment == 'S0']
    
    nbp_drift = get_nbp_drift(s0_df)
    nbp_offset = abs(s0_df.NBP.mean())

    if nbp_drift < drift_tol and nbp_offset < offset_tol:
        print("S0 simulation passes TRENDY equilibrium criteria!")
    else: 
        print("S0 simulation DOES NOT pass TRENDY equilibrium criteria!")
    
    print(" ")
    print("-------------------------------------------------------------")
    print(" ")
    print("Drift is", round(nbp_drift, 4), "PgC/yr/century",
          "and offset is", round(nbp_offset, 4), "PgC/yr")
    
def calculate_global_mean(da: xr.DataArray) -> xr.DataArray:
    """Calculates global mean for a variable

    Args:
        da (xr.DataArray): input DataArray

    Returns:
        xr.DataArray: output DataArray
    """
    
    global_mean = da.mean(dim=['lat', 'lon'])
    
    return global_mean
  
def get_data_dict(var_df: pd.DataFrame, var_out: str) -> dict:
    """Creates a dictionary with TRENDY-relevant information about a TRENDY output 
        variable

    Args:
        var_df (pd.DataFrame): pandas DataFrame with information about each TRENDY variable
        var_out (str): output variable

    Returns:
        dict: dictionary with TRENDY-relevant information about a TRENDY output variable
    """

    df_sub = var_df[var_df['output_varname'] == var_out]

    data_dict = {
      'vars_in': df_sub['CLM_varname'].values[0].split('-'),
      'long_name': df_sub['long_name'].values[0],
      'output_units': df_sub['output_units'].values[0],
      'fill_val': df_sub['NA_val'].values[0],
      'conversion_factor': df_sub['conversion_factor'].values[0],
      'global_cf': df_sub['global_cf'].values[0],
      'frequency': df_sub['frequency'].values[0],
      'dims': df_sub['CMOR_dims'].values[0],
      'dimension': df_sub['dimension'].values[0],
      'output_function': df_sub['output_function'].values[0]
    }

    if data_dict['dimension'] == 'pft':
        data_dict['stream'] = 'h1'
    else:
        data_dict['stream'] = 'h0'

    return data_dict

def get_var_files(tseries_dir: str, stream: str, var: str) -> list[str]:
    """Returns a list of files for a variable and stream

    Args:
        tseries_dir (str): path to single-variable time-series folder
        stream (str): stream [h0, h1]
        var (str): variable

    Returns:
        list[str]: list of files
    """

    var_files = sorted(glob.glob(f'{tseries_dir}/*{stream}.{var}.*'))

    return var_files

def preprocess(ds: xr.Dataset, data_vars: list[str]) -> xr.Dataset:
    """Preprocess xarray dataset to only read in some variables

    Args:
        ds (xr.Dataset): xarray dataset
        data_vars (list[str]): list of data variables to read in

    Returns:
        xr.Dataset: output dataset
    """
    
    return ds[data_vars]

def read_in_ds(files: list[str], vars: list[str], stream: str, min_year: int=1700,
               max_year: int=2023) -> xr.Dataset:
    """Reads in and concatenates a dataset from a list of files, also fixes the 
        time variable to fix the year 0 issue in CLM

    Args:
        files (list[str]): list of files
        vars (list[str]): list of variables to read in
        stream (str): file stream [h0, h1]
        min_year (int, optional): minimum year to subset dataset to. Defaults to 1700.
        max_year (int, optional): maximum year to subset dataset to. Defaults to 2023.

    Returns:
        xr.Dataset: output dataset
    """

    with dask.config.set(**{'array.slicing.split_large_chunks': False}):
        ds = xr.open_mfdataset(files, combine='nested', concat_dim='time', parallel=True,
                               autoclose=True, 
                               preprocess=functools.partial(preprocess, data_vars=vars),
                               chunks={'time': 10})
    
    # fix time problem
    year_zero = ds['time.year'][0].values
    ds['time'] = xr.cftime_range(str(year_zero), periods=len(ds.time), freq='MS',
                                 calendar='noleap')
    
    if stream == 'h1':
        # add in these variables for h1 streams
        save_vars = ['lat', 'lon', 'pfts1d_itype_veg', 'pfts1d_ixy', 'pfts1d_jxy',
                     'landfrac']
        tmp = xr.open_dataset(files[0])
        for var in save_vars:
            ds[var] = tmp[var]
    
    # subset to min/max time
    ds = ds.sel(time=slice(f'{min_year}-01-01', f'{max_year}-12-31'))
    
    return ds

def get_base_da(ds: xr.Dataset, pft: int) -> xr.DataArray:
    """Creates a base xarray DataArray for pft regridding

    Args:
        ds (xr.Dataset): input dataset 
        pft (int): pft index we are regridding

    Returns:
        xr.DataArray: DataArray with correct dimensions, filled with NaNs
    """

    nlat = len(ds.lat)
    nlon = len(ds.lon)
    ntime = len(ds.time)
    
    nan = xr.DataArray(np.zeros([nlat, nlon, 1, ntime]) + np.nan,
                       dims=['lat', 'lon', 'PFT', 'time'])
    nan['lat'] = ds.lat
    nan['lon'] = ds.lon
    nan['PFT'] = [pft]

    return nan

def create_regridder(ds: xr.Dataset, nlon: int, pft: int):
    """Creates regridder DataArrays for regridding an input Dataset to a specific pft

    Args:
        ds (xr.Dataset): input dataset
        nlon (int): number of longitudes on dataset
        pft (int): pft index to regrid to

    Returns:
        xr.DataArray: regridder DataArray
        xr.DataArray: DataArray masked to the input pft
    """

    ixpft = ds.pfts1d_itype_veg == pft
    ixy = ds.pfts1d_ixy.isel(pft=ixpft) - 1
    jxy = ds.pfts1d_jxy.isel(pft=ixpft) - 1
    regridder = (jxy*nlon + ixy).astype(int).compute()

    return regridder, ixpft

def regrid_h1_ds(ds: xr.Dataset, pft: int, var: str, pft_names: xr.DataArray) -> xr.Dataset:
    """Regrids an h1 dataset and variable to a specific pft

    Args:
        ds (xr.Dataset): input Dataset
        pft (int): pft index to regrid to
        var (str): CLM history variable to regrid
        pft_names (xr.DataArray): DataArray of pft names (indexed by PFT index)

    Returns:
        xr.Dataset: output Dataset
    """
    
    # create regridder
    nan = get_base_da(ds, pft)
    regridder, ixpft = create_regridder(ds, len(nan.lon), pft)

    # create regridded variable array
    var_array = nan.copy(deep=True).stack({'gridcell': ['lat', 'lon']})
    var_array[0, :, regridder] = ds[var].isel(pft=ixpft)
    da = var_array.unstack()
    da.attrs = ds[var].attrs
    
    # create dataset
    out = xr.Dataset()
    if var == 'pfts1d_wtgcell':
        out[var] = (ds.landfrac*da)
    else:
        out[var] = da
    
    out['time'] = ds.time
    out['pftname'] = xr.DataArray([pft_names[pft].values], dims='PFT')
    
    return out
    
def read_in_all_vars(vars_in: list[str], tseries_dir: str, stream: str,
                     pft_names: xr.DataArray, pft: int=None):
    """Read in and merges CLM history variables in input list
        If the variable is an h1 stream, the data is regridded to the input pft

    Args:
        vars_in (list[str]): list of CLM history variables
        tseries_dir (str): path to single-variable time-series folder
        stream (str): file stream [h0, h1]
        pft_names (xr.DataArray): DataArray of pft names (indexed by PFT index)
        pft (int, optional): PFT index to regrid to if an h1 stream. Defaults to None.

    Raises:
        ValueError: Trying to read in an h1 stream but no pft index was supplied

    Returns:
        xr.Dataset: output Dataset
    """
    
    ds_list = []
    for var in vars_in:
        
        if var in ['FLDS', 'FSDS']:
            # these variables don't exist on the h1 stream but we do use them to 
            # calculate pft-level output
            var_stream = 'h0'
        else:
            var_stream = stream
                
        # get files for this variable
        files = get_var_files(tseries_dir, var_stream, var)
        
        # read in the dataset
        ds_var = read_in_ds(files, [var], var_stream)
        
        if var_stream == 'h1':
            if pft is None:
                raise ValueError('Must supply a pft for h1 streams.')
        
            ds_regrid = regrid_h1_ds(ds_var, pft, var, pft_names)
            ds_list.append(ds_regrid)
        else:
            ds_list.append(ds_var)
    
    # merge together
    ds = xr.merge(ds_list)

    return ds

def sum_vars(ds: xr.Dataset, vars_in: list[str], var_out: str):
    """sum up variables in input list

    Args:
        ds (xr.Dataset): input Dataset
        vars_in (list[str]): list of variables to sum
        var_out (str): name of new DataArray

    Returns:
        xr.Dataset: output Dataset
        str: string explaining how this variable was calculated
    """

    # create an empty data array
    ds[var_out] = xr.full_like(ds[vars_in[0]], 0.0)
    
    # sum up input variables
    for var in vars_in:
        ds[var_out] += ds[var]

    # create string to explain how this variable was calculated
    clm_orig_var_string = ' + '.join(vars_in)

    return ds, clm_orig_var_string

def find_cpool_var(vars_in: list[str], cpool_string: str) -> str:
    """Finds the matching CLM history variable for the correct carbon pool

    Args:
        vars_in (list[str]): input list of history variables
        cpool_string (str): carbon pool string to match (i.e. ACT, PAS, SLO)

    Returns:
        str: CLM history variable name
    """
    
    return vars_in[np.argwhere([cpool_string in var for var in vars_in])[0][0]]

def reorg_by_cpool(ds: xr.Dataset, var_out: str, active_var: str, slow_var: str,
                   passive_var: str) -> xr.Dataset:
    """Reorganizes a dataset to be indexed by carbon pool

    Args:
        ds (xr.Dataset): input Dataset
        var_out (str): output variable name
        active_var (str): name of DataArray for active carbon pool
        slow_var (str): name of DataArray for slow carbon pool
        passive_var (str): name of DataArray for passive carbon pool

    Returns:
        xr.Dataset: output Dataset
    """
    
    active = ds[active_var].to_dataset(name=var_out)
    slow = ds[slow_var].to_dataset(name=var_out)
    passive = ds[passive_var].to_dataset(name=var_out)

    ds_pools = xr.concat([active, slow, passive], dim='Pool', data_vars='all')
    ds_pools = ds_pools.assign_coords(Pool=("Pool", ['active', 'slow', 'passive']))
    ds_pools = ds_pools.transpose('time', 'Pool', 'lat', 'lon')

    return ds_pools

def carbon_pool(ds: xr.Dataset, vars_in: list[str], var_out: str):
    """Calculates TRENDY's cSoilpools variable (Carbon in individual soil pools)

    Args:
        ds (xr.Dataset): input Dataset
        vars_in (list[str]): input CLM variables needed for this output
        var_out (str): output variable name

    Returns:
        xr.Dataset: output Dataset
        str: string explaining how this variable was calculated
    """

    # find matching strings
    active_var = find_cpool_var(vars_in, "ACT")
    slow_var = find_cpool_var(vars_in, "SLO")
    passive_var = find_cpool_var(vars_in, "PAS")
    
    # reorganize by carbon pool
    ds = reorg_by_cpool(ds, var_out, active_var, slow_var, passive_var)

    clm_orig_var_string = ', '.join(vars_in)

    return ds, clm_orig_var_string

def rh_pool(ds: xr.Dataset, vars_in: list[str], var_out: str):
    """Calculates TRENDY's rhpool variable (Carbon Flux from individual soil pools)

    Args:
        ds (xr.Dataset): input Dataset
        vars_in (str[list]): int CLM variables needed for this output
        var_out (str): output variable name

    Returns:
        xr.Dataset: output Dataset
        str: string explaining how this variable was calculated
    """
    
    # calculate each variable
    ds['FROM_ACT'] = ds['SOM_ACT_C_TO_SOM_PAS_C'] + ds['SOM_ACT_C_TO_SOM_SLO_C']
    ds['FROM_SLO'] = ds['SOM_SLO_C_TO_SOM_ACT_C'] + ds['SOM_SLO_C_TO_SOM_PAS_C']
    ds['FROM_PAS'] = ds['SOM_PAS_C_TO_SOM_ACT_C']

    # reorganize by carbon pool
    ds_pools = reorg_by_cpool(ds, var_out, 'FROM_ACT', 'FROM_SLO', 'FROM_PAS')

    clm_orig_var_string = 'active: SOM_ACT_C_TO_SOM_PAS_C, SOM_ACT_C_TO_SOM_SLO_C; slow: SOM_SLO_C_TO_SOM_ACT_C, SOM_SLO_C_TO_SOM_PAS_C; passive: SOM_PAS_C_TO_SOM_ACT_C'

    return ds_pools, clm_orig_var_string

def black_sky_albedo(ds: xr.Dataset, vars_in: list[str], var_out: str):
    """Calculates black sky albedo

    Args:
        ds (xr.Dataset): input Dataset
        vars_in (str[list]): int CLM variables needed for this output
        var_out (str): output variable name

    Returns:
        xr.Dataset: output Dataset
        str: string explaining how this variable was calculated
    """

    bs_alb = (ds['FSRND'] + ds['FSRVD'])/(ds['FSDSND'] + ds['FSDSVD'])
    data_out = bs_alb.to_dataset(name=var_out)
    data_out['pftname'] = ds.pftname

    clm_orig_var_string = '(FSRND + FSRVD)/(FSDSND + FSDSVD)'

    return data_out, clm_orig_var_string

def white_sky_albedo(ds: xr.Dataset, vars_in: list[str], var_out: str):
    """Calculates white sky albedo

    Args:
        ds (xr.Dataset): input Dataset
        vars_in (str[list]): int CLM variables needed for this output
        var_out (str): output variable name

    Returns:
        xr.Dataset: output Dataset
        str: string explaining how this variable was calculated
    """

    ws_alb = (ds['FSRNI'] + ds['FSRVI'])/(ds['FSDSNI'] + ds['FSDSVI'])
    data_out = ws_alb.to_dataset(name=var_out)
    data_out['pftname'] = ds.pftname

    clm_orig_var_string = '(FSRNI + FSRVI)/(FSDSNI + FSDSVI)'
    
    return data_out, clm_orig_var_string

def calculate_annual_mean(da: xr.DataArray) -> xr.DataArray:
    """Calculates the annual mean for a DataArray

    Args:
        da (xr.DataArray): input DataArray

    Returns:
        xr.DataArray: output DataArray
    """

    months = da['time.daysinmonth']
    annual_mean = 1/365*(months*da).groupby('time.year').sum()
    annual_mean.name = da.name
    annual_mean.attrs = da.attrs
    
    return annual_mean

def make_annual(ds: xr.Dataset, var_out: str, fill_val: float) -> xr.Dataset:
    """Calculates an annual dataset 

    Args:
        ds (xr.DataSet): input Dataset
        var_out (str): output variable name
        fill_val (float): fill value for new annual variable

    Returns:
        xr.Dataset: output Dataset
    """

    # calculate annual mean
    annual_mean = calculate_annual_mean(ds[var_out])
    data_out = annual_mean.to_dataset(name=var_out)
    
    # rename year to time as per TRENDY instructions
    data_out = data_out.rename({'year': 'time'})
    data_out['time'].attrs = {
        'long_name': 'year',
        'units': 'yr'
    }

    return data_out

def create_trendy_var(file_name: str, data_dict: dict, func_dict: dict, var_out: str, tseries_dir: str, pft_names: xr.DataArray, pft :int=None):
    """Creates and writes out a TRENDY variable based on information in the data_dict and func_dict

    Args:
        file_name (str): full path of file name to write to
        data_dict (dict): dictionary with TRENDY-relevant information about a TRENDY output variable
        func_dict (dict): dictionary mapping to functions used to calculate output  variables
        var_out (str): output variable name
        tseries_dir (str): path to single-variable time-series folder
        pft_names (xr.DataArray): DataArray of pft names (indexed by PFT index)
        pft (int, optional): PFT index to regrid to for h1 streams. Defaults to None.

    Raises:
        KeyError: Can't find function in dictionary
        ValueError: Unknown frequency for variable
        ValueError: Unknown dimension for variable
    """
    
    # CLM history variables
    vars_in = data_dict['vars_in']
    
    # read in all variables required for this output variable
    # if an h1 stream, it is also regridded to the input pft
    ds = read_in_all_vars(vars_in, tseries_dir, data_dict['stream'], pft_names, pft)
    
    # save info for later
    units = {}
    long_names = {}
    for var in vars_in:
        units[var] = ds[var].attrs['units']
        long_names[var] = ds[var].attrs['long_name']
        
    # create new variable based on input function
    try:
        ds, func_string = func_dict[data_dict['output_function']](ds, vars_in, var_out)
    except KeyError:
        print(f'Unknown function for variable {var_out}')
        raise
    
    # convert to correct units

    if var_out == 'burntArea':
        months = ds['time.daysinmonth']
        fractional = (months*ds[var_out]*24*60*60)
        ds[var_out] = fractional
    else:
        ds[var_out] = ds[var_out]*data_dict['conversion_factor']
        
    # convert to annual if required
    if data_dict['frequency'] == 'annual':
        data_out = make_annual(ds, var_out, data_dict['fill_val'])
    elif data_dict['frequency'] == 'monthly':
        data_out = ds[var_out].to_dataset(name=var_out)
    else:
        raise ValueError(f'Unknown frequency for variable {var_out}')
    
    # rename dimensions for soil layers
    if data_dict['dimension'] == 'soil_layer':
        if var_out == 'tsl':
            data_out = data_out.rename({'levgrnd': 'stlayer'})
        elif var_out == 'msl':
            data_out = data_out.rename({'levsoi': 'smlayer'})
        else:
            raise ValueError(f'unknown output dimension for variable {var_out}')
        
    # set attributes
    data_out[var_out].attrs = {
        'long_name': data_dict['long_name'],
        'units': data_dict['output_units'],
        '_FillValue': data_dict['fill_val'],
        'CLM-TRENDY_unit_conversion_factor': data_dict['conversion_factor'],
        'CLM_orig_var_name': func_string
    }
    if var_out == 'burntArea':
        data_out[var_out].attrs['CLM-TRENDY_unit_conversion_factor'] = 'days_in_month*24*60*60'
    for var in vars_in:
        data_out[var_out].attrs[f"CLM_orig_attr_{var}_units"] = units[var]
        data_out[var_out].attrs[f"CLM_orig_attr_{var}_long_name"] = long_names[var]
        
    # save h1 stream data and transpose
    if data_dict['stream'] == 'h1':
        data_out['pftname'] = ds.pftname
        data_out['pftname'].attrs = {'long_name': 'pft name'}
        data_out['PFT'].attrs = {'long_name': 'pft index'}
        data_out = data_out.transpose('time', 'PFT', 'lat', 'lon')
        
    # fill na values
    data_out = data_out.fillna(data_dict['fill_val'])
    
    encoding = {
        var_out: {'dtype': 'float32'},
        'lat': {
            'dtype': 'float32',
            '_FillValue': -9999.0
        },
        'lon': {
            'dtype': 'float32',
            '_FillValue': -9999.0
        },
        'time': {'dtype': 'int32'}
    }
    
    if data_dict['stream'] == 'h1':
        
        encoding['PFT'] = {'dtype': 'int32',
                           '_FillValue': -9999}
        encoding['pftname'] = {'dtype': '<U36'}
        
        data_out.to_netcdf(file_name, unlimited_dims='PFT', encoding=encoding)
    else:
        data_out.to_netcdf(file_name, encoding=encoding)

def create_all_trendy_vars(out_dir: str, var_df_file: str, exp: str, tseries_dir: str, pft_list: list[int], clm_param_file: str, clobber :bool=False):
    """Create all (except two)

    Args:
        out_dir (str): path to output directory
        var_df_file (str): file name describing TRENDY variables to create
        exp (str): TRENDY experiment [S0, S1,...]
        tseries_dir (str): top directory where time series data are located
        pft_list (list[int]): list of PFTs for this simulation
        clm_param_file (str): path to CLM parameter file
        clobber (bool, optional): whether or not to overwrite data. Defaults to False.
    """
    
    # dictionary for calculating output variables
    funcs = {
        'sum': sum_vars,
        'carbon_pool': carbon_pool,
        'rh_pool': rh_pool,
        'black_sky_albedo': black_sky_albedo,
        'white_sky_albedo': white_sky_albedo
    }
    
    # information about postprocessing
    var_df = pd.read_csv(var_df_file)
    
    skip_vars = ['oceanCoverFrac', 'landCoverFrac']
    
    all_vars = [var_out for var_out in var_df['output_varname'].values if var_out not in skip_vars]
        
    clm_param = xr.open_dataset(clm_param_file)
    pft_names = xr.DataArray([str(pft)[2:-1].strip() for pft in clm_param.pftname.values],
                             dims='PFT')
        
   # loop through all output variables
    for var_out in all_vars:
         
        # grab data dictionary for this output variable
        data_dict = get_data_dict(var_df, var_out)
        
        if data_dict['stream'] == 'h1':
            for pft in pft_list:
                file_name = os.path.join(out_dir, 'to_trendy', exp,
                                         f"CLM6.0_{exp}_{var_out}_PFT{pft:02}.nc")
                if os.path.isfile(file_name) and not clobber:
                    print(f'Skipping {var_out} for pft {pft}')
                    continue
                else:
                    if os.path.isfile(file_name):
                        os.remove(file_name)
                    create_trendy_var(file_name, data_dict, funcs, var_out, tseries_dir,
                                      pft_names, pft)
        else:
            file_name = os.path.join(out_dir, 'to_trendy', exp, f"CLM6.0_{exp}_{var_out}.nc")
            if os.path.isfile(file_name) and not clobber:
                print(f'Skipping {var_out}')
                continue
            else:
                if os.path.isfile(file_name):
                    os.remove(file_name)
                create_trendy_var(file_name, data_dict, funcs, var_out, tseries_dir,
                                  pft_names)
            
def regrid_all_landcoverfracs(hist_dir: str, exp: str, var_df_file: str, 
                              clm_param_file: str, var_out: str, pft_list: list[int], 
                              out_dir: str):
    """Creates and writes out the TRENDY landCoverFrac variable for all PFTs

    Args:
        hist_dir (str): path to history directory for this TRENDY experiment
        exp (str): TRENDY experiment name [S0, S1,...]
        var_df_file (str): file name describing TRENDY variables to create
        clm_param_file (str): path to CLM parameter file
        var_out (str): output variable name
        pft_list (list[int]): list of pfts to convert
        out_dir (str): output directory
    """
    
    # information about postprocessing
    var_df = pd.read_csv(var_df_file)
    
    clm_param = xr.open_dataset(clm_param_file)
    pft_names = xr.DataArray([str(pft)[2:-1].strip() for pft in clm_param.pftname.values],
                             dims='PFT')
    
    # grab data dictionary for this output variable
    data_dict = get_data_dict(var_df, var_out)
    var_in = data_dict['vars_in'][0]
    
    # get land coverfraction for all pfts
    ds = read_landcoverfrac_ds(hist_dir, exp, var_in)
    
    # regrid each pft
    for pft in pft_list:
    
        file_name = os.path.join(out_dir, 'to_trendy', exp, f"CLM6.0_{exp}_{var_out}_PFT{pft:02}.nc")
        
        if not os.path.isfile(file_name):
            data_out = regrid_landcoverfrac(ds, var_out, var_in, data_dict, pft, pft_names)
            encoding = {
                    var_out: {'dtype': 'float32'},
                    'lat': {
                        'dtype': 'float32',
                        '_FillValue': -9999.0
                    },
                    'lon': {
                        'dtype': 'float32',
                        '_FillValue': -9999.0
                    },
                    'time': {'dtype': 'int32'},
                'PFT': {'dtype': 'int32',
                        '_FillValue': -9999},
                'pftname': {'dtype': '<U36'}
                }
            data_out.to_netcdf(file_name, unlimited_dims='PFT', encoding=encoding)
        else:
            print(f"Skipping land cover frac for pft {pft}...")

def create_oceanfrac(hist_dir, exp, var_df_file, var_out, out_dir, clobber=False):

    file_name = os.path.join(out_dir, 'to_trendy', exp, f"CLM6.0_{exp}_{var_out}.nc")
    
    if os.path.isfile(file_name) and not clobber:
        print(f'Skipping {var_out}')
        return
    
    if os.path.isfile(file_name):
        os.remove(file_name)
    
    ds = get_land_area(hist_dir)
    ds[var_out] = 1.0 - ds.landfrac
    data_out = ds[var_out].to_dataset(name=var_out)
    
    # information about postprocessing
    var_df = pd.read_csv(var_df_file)
    
    # grab data dictionary for this output variable
    data_dict = get_data_dict(var_df, var_out)
    
    # set attributes
    data_out[var_out].attrs = {
        'long_name': data_dict['long_name'],
        'units': '',
        '_FillValue': data_dict['fill_val'],
        'CLM-TRENDY_unit_conversion_factor': data_dict['conversion_factor'],
        'CLM_orig_var_name': '1.0 - landfrac'
    }
    data_out[var_out].attrs["CLM_orig_attr_landfrac_long_name"] = ds.landfrac.attrs['long_name']
    data_out = data_out.fillna(data_dict['fill_val'])
    
    encoding = {
        var_out: {'dtype': 'float32'},
        'lat': {
            'dtype': 'float32',
            '_FillValue': -9999.0
        },
        'lon': {
            'dtype': 'float32',
            '_FillValue': -9999.0
        },
    }
    
    data_out.to_netcdf(file_name, encoding=encoding)

def create_gridarea(hist_dir, exp, var_out, out_dir, clobber=False):

    file_name = os.path.join(out_dir, 'to_trendy', exp, f"CLM6.0_{exp}_{var_out}.nc")
    
    if os.path.isfile(file_name) and not clobber:
        print(f'Skipping {var_out}')
        return
    
    if os.path.isfile(file_name):
        os.remove(file_name)
    
    ds = get_land_area(hist_dir)
    ds[var_out] = ds.area
    data_out = ds[var_out].to_dataset(name=var_out)

    # set attributes
    data_out[var_out].attrs = {
        'long_name': 'gridcell area',
        'units': 'm2',
        '_FillValue': -9999.0,
        'CLM_orig_var_name': 'area',
        'CLM-TRENDY_unit_conversion_factor': 1e6
    }
    data_out[var_out].attrs["CLM_orig_attr_area_units"] = 'km2'
    data_out[var_out].attrs["CLM_orig_attr_area_long_name"] = ds.area.attrs['long_name']
    
    data_out = data_out.fillna(-9999.0)
    
    encoding = {
        var_out: {'dtype': 'float32'},
        'lat': {
            'dtype': 'float32',
            '_FillValue': -9999.0
        },
        'lon': {
            'dtype': 'float32',
            '_FillValue': -9999.0
        },
    }
    
    data_out.to_netcdf(file_name, encoding=encoding)
