"""Methods for post-processing TRENDY files"""
import os
import glob
import math
import functools
import dask
import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# latest year - UPDATE THIS EACH YEAR
LATEST_TRENDY_YEAR = 2024

# stream names
GLOBAL_STREAM = "h0"
PFT_STREAM = "h1"
DAILY_STREAM = "h2"

# encoding constants
NETCDF_ENCODING = {
    "lat": {"dtype": "float32", "_FillValue": -9999.0},
    "lon": {"dtype": "float32", "_FillValue": -9999.0},
    "time": {"dtype": "int32"},
    "PFT": {"dtype": "int32", "_FillValue": -9999},
    "pftname": {"dtype": "<U36"},
}

# dimension renaming rules for soil-layered variables
DIM_RENAME_MAP = {
    "tsl": {"levgrnd": "stlayer"},
    "msl": {"levsoi": "smlayer"},
}


def get_active_pfts(hist_dir: str, stream: str = "h1") -> list[int]:
    """Gets list of pft indices that are active for a specific archived history output

    Args:
        hist_dir (str): path to history directory
        stream (str, optional): h stream. Defaults to 'h1'.

    Returns:
        list[int]: list of pft indices active for this run
    """

    pft_files = sorted(glob.glob(f"{hist_dir}/*.{stream}*.nc"))

    ds = xr.open_dataset(pft_files[0])
    active_pfts = [int(pft) for pft in np.unique(ds.pfts1d_itype_veg)]

    return active_pfts


def calculate_global_sum(
    da: xr.DataArray, land_area: xr.DataArray = None
) -> xr.DataArray:
    """Calculates global sum for a variable, multiplies by land area if supplied

    Args:
        da (xr.DataArray): input DataArray
        land_area (xr.DataArray): land area DataArray, defaults to None

    Returns:
        xr.DataArray: output DataArray
    """

    if land_area is not None:
        global_sum = (da * land_area).sum(dim=["lat", "lon"])
    else:
        global_sum = da.sum(dim=["lat", "lon"])

    return global_sum


def calculate_global_mean(
    da: xr.DataArray, land_area: xr.DataArray = None
) -> xr.DataArray:
    """Calculates global sum for a variable, multiplies by land area if supplied

    Args:
        da (xr.DataArray): input DataArray
        land_area (xr.DataArray): land area DataArray, defaults to None

    Returns:
        xr.DataArray: output DataArray
    """

    if land_area is not None:
        global_sum = (da * land_area).sum(dim=["lat", "lon"])
    else:
        global_sum = da.sum(dim=["lat", "lon"])

    return global_sum

def get_pft_files(hist_dir: str, exp: str, stream: str) -> list[str]:
    """Returns a list of PFT files from an input history directory and TRENDY experiment

    Args:
        hist_dir (str): path to history diretory
        exp (str): TRENDY experiment [S0, S1, ...]
        stream (str): stream name

    Returns:
        list[str]: sorted list of files
    """
    return sorted(
        glob.glob(os.path.join(hist_dir, f"*{exp}*.clm2.{stream}a.*.nc"))
    )


def read_landcoverfrac_ds(hist_dir: str, exp: str, var_in: str) -> xr.Dataset:
    """Read CLM history files for a TRENDY experiment and extract data needed for landCoverFrac regridding.

    Args:
        hist_dir (str): path to history directory
        exp (str): TRENDY experiment name [S0, S1,...]
        var_in (str): variable to convert

    Returns:
        xr.Dataset: output dataset
    Raises:
        FileNotFoundError: couldn't find PFT files
    """

    # find the pft archive files
    pft_files = get_pft_files(hist_dir, exp, PFT_STREAM)
    if len(pft_files) == 0:
        raise FileNotFoundError(f"No {PFT_STREAM} files found in {hist_dir} for experiment {exp}.")

    # open the dataset
    ds = xr.open_mfdataset(
        pft_files,
        combine="nested",
        concat_dim="time",
        parallel=True,
        autoclose=True,
        preprocess=functools.partial(preprocess, data_vars=[var_in, "time"]),
        chunks={"time": 10},
    )

    # add in these variables for PFT streams
    static_vars = [
        "lat",
        "lon",
        "pfts1d_itype_veg",
        "pfts1d_ixy",
        "pfts1d_jxy",
        "landfrac",
    ]
    with xr.open_dataset(pft_files[0]) as tmp:
        for var in static_vars:
            if var not in ds:
                ds[var] = tmp[var]

    return ds


def regrid_landcoverfrac(
    ds: xr.Dataset,
    var_out: str,
    var: str,
    data_dict: dict,
    pft: int,
    pft_names: xr.DataArray,
) -> xr.Dataset:
    """Regrids CLM PFT land cover fraction for a specific PFT and returns it as an annual-mean dataset.

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
    
    # regrid dataset to lat/lon space for this PFT
    ds_regridded = regrid_pft_ds(ds, pft, var, pft_names)
    
    # compute annual means of regridded data
    ds_annual = calculate_annual_mean(ds_regridded.pfts1d_wtgcell)
    data_out = ds_annual.to_dataset(name=var_out)

    # add time metadata
    data_out = data_out.rename({"year": "time"})
    data_out.time.attrs = {"long_name": "years since 1700"}

    # set attributes
    data_out[var_out].attrs = {
        "long_name": data_dict["long_name"],
        "units": data_dict["output_units"],
        "_FillValue": data_dict["fill_val"],
        "CLM-TRENDY_unit_conversion_factor": data_dict["conversion_factor"],
        "CLM_orig_var_name": var,
        f"CLM_orig_attr_{var}_long_name": ds["pfts1d_wtgcell"].attrs.get("long_name", "")
    }
    
    # arrange dimensions and add PFT metadata
    data_out = data_out.transpose("time", "PFT", "lat", "lon")
    data_out["pftname"] = ds_regridded.pftname
    data_out["pftname"].attrs = {"long_name": "pft name"}
    data_out["PFT"].attrs = {"long_name": "pft index"}
    
    # fill any remaining missing values
    data_out = data_out.fillna(data_dict["fill_val"])

    return data_out


def get_land_area(hist_dir: str) -> xr.DataArray:
    """Grabs land area and land fraction for a CLM case

    Args:
        hist_dir (str): path to history directory

    Returns:
        xr.DataArray: DataArray of grid area [m2] and grid fraction
    """

    # open a file to get land area
    hist_file = sorted(glob.glob(f"{hist_dir}/*{GLOBAL_STREAM}*"))[0]
    ds = xr.open_dataset(hist_file)

    # get gridcell area and land area
    grid_area = 1e6 * ds.area
    grid_area.attrs = {"long_name": "grid cell areas", "units": "m2"}

    land_area = xr.Dataset()
    land_area["area"] = grid_area
    land_area["landfrac"] = ds.landfrac

    return land_area


def round_up(n: float, decimals: int = 0) -> int:
    """Helper function to round up a number to a specified number of integers

    Args:
        n (float): input number
        decimals (int, optional): Number to round to. Defaults to 0.

    Returns:
        int: rounded number
    """

    multiplier = 10**decimals
    return math.ceil(n * multiplier) / multiplier


def truncate(n: float, decimals: int = 0) -> int:
    """Helper function to round down a number to a specified number of integers

    Args:
        n (float): input number
        decimals (int, optional): Number to round to. Defaults to 0.

    Returns:
        int: rounded number
    """

    multiplier = 10**decimals
    return int(n * multiplier) / multiplier


def make_nbp_df(top_dir: str, exp: str):
    """Creates and writes a Net Biome Productivity (NBP) dataset summary by region for a 
        TRENDY experiment.

    Args:
        top_dir (str): path to directory containing CLM6.0 experiment output files
        exp (str): TRENDY experiment name (e.g., S0, S1, etc.)
    """

    # load input datasets
    nbp_file = os.path.join(top_dir, f"CLM6.0_{exp}_nbp.nc")
    grid_area_file = os.path.join(top_dir, f"CLM6.0_{exp}_gridArea.nc")
    ocean_frac_file = os.path.join(top_dir, f"CLM6.0_{exp}_oceanCoverFrac.nc")
    
    nbp_ds = xr.open_dataset(nbp_file)
    grid_area = xr.open_dataset(grid_area_file)
    ocean_frac = xr.open_dataset(ocean_frac_file)
    
    # compute land area
    land_frac = 1.0 - ocean_frac.oceanCoverFrac
    land_area_global = grid_area.gridArea * land_frac

    # define regional masks
    lats = land_area_global.lat
    regions = {
        "Global": land_area_global,
        "North": land_area_global.where(lats > 30),
        "Tropics": land_area_global.where(abs(lats) < 30),
        "South": land_area_global.where(lats < -30),
    }
    
    # convert NBP units 
    conversion_factor = 1e-12 * 24 * 60 * 60 * 365 
    
    # compute NBP time series for each region
    nbp_by_region = {region: compute_nbp_region(nbp_ds.nbp, area, conversion_factor) for region, area in regions.items()}

    # construct output dataframe
    years = nbp_by_region['Global'].year.values
    df = pd.DataFrame({region: nbp_by_region[region] for region in nbp_by_region}, index=years)
    df.index.name = "Year"
    
    # write to CSV
    out_file = os.path.join(top_dir, f"CLM6.0_{exp}_zonalNBP.txt")
    df.to_csv(out_file)


def compute_nbp_region(da: xr.DataArray, region_area: xr.DataArray, conversion_factor: float) -> xr.DataArray:
    """Calculate total net biome production for a region of the globe

    Args:
        da (xr.DataArray): global net biome production
        region_area (xr.DataArray): region mask
        conversion_factor (float): conversion factor NBP

    Returns:
        xr.DataArray: total NBP
    """
    nbp_annual = calculate_annual_mean(da)
    return conversion_factor * calculate_global_sum(nbp_annual, region_area)

def get_all_nbp_df(out_dir: str, exps: list[str]) -> pd.DataFrame:
    """Reads and combines zonal NBP dataframes from multiple TRENDY experiments.

    Args:
        out_dir (str): path to the base output directory containing experiment subfolders
        exps (list[str]): list of experiments to include

    Returns:
        pd.DataFrame: Combined DataFrame with columns: Year, experiment, region, NBP.
    """

    data_frames = []
    for exp in exps:
        file_in = os.path.join(out_dir, exp, f"CLM6.0_{exp}_zonalNBP.txt")
        df = pd.read_csv(file_in)
        df["experiment"] = exp
        data_frames.append(df)

    # concatenate and reshape to long format
    nbp = pd.concat(data_frames, ignore_index=True)
    nbp_long = nbp.melt(
        id_vars=["Year", "experiment"],
        value_vars=["Global", "North", "Tropics", "South"],
        value_name="NBP",
        var_name="region",
    )

    return nbp_long

def get_blank_plot(width=7, height=5):
    """Generates a blank plot"""

    plt.figure(figsize=(width, height))
    ax = plt.subplot(111)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()


def plot_running_means(df: pd.DataFrame):
    """Plots 20-year running means of global NBP for each TRENDY experiment.

    Args:
        df (pd.DataFrame): input DataFrame with columns: 'Year', 'NBP', 'experiment'.
    """
    
    # determine y-axis limits
    minval = df["NBP"].min()
    minvar = -round_up(abs(minval)) if minval < 0 else truncate(minval)
    maxvar = round_up(df["NBP"].max())

    # tableau color palette normalized to [0, 1]
    tableau = [
        (31, 119, 180),
        (255, 127, 14),
        (44, 160, 44),
        (214, 39, 40),
    ]
    colors = [(r / 255.0, g / 255.0, b / 255.0) for r, g, b in tableau]

    # set up plot
    get_blank_plot()
    plt.xlim([1720, LATEST_TRENDY_YEAR])
    plt.ylim(minvar, maxvar)
    plt.grid(True, which="both", axis="y", linestyle="--", linewidth=0.5, color="black", alpha=0.3)
    plt.tick_params(bottom=False, top=False, left=False, right=False)

    # plot rolling means
    for i, exp in enumerate(sorted(np.unique(df.experiment))):
        subset = df[df["experiment"] == exp]
        rolling_mean = subset['NBP'].rolling(window=20, center=False).mean()
        plt.plot(subset['Year'], rolling_mean, label=exp, color=colors[i])

    # reference lines
    plt.axhline(y=0.1, color="gray", linestyle=":")
    plt.axhline(y=-0.1, color="gray", linestyle=":")
    
    # labels
    plt.xlabel("Year")
    plt.ylabel("NBP 20-yr running mean (PgC yr$^{-1}$)")
    plt.legend(loc="upper left")


def plot_cumulative_nbp(df: pd.DataFrame):
    """Plots cumulative NBP for each region and TRENDY experiment.

    Args:
        df (pd.DataFrame): input DataFrame with columns 'Year', 'NBP', 'region', 'experiment'.
    """
    regions = sorted(df["region"].unique())
    experiments = sorted(df["experiment"].unique())
    
    _, axes = plt.subplots(2, 2, figsize=(8, 6), sharex=True)
    axes = axes.flatten()
    
    for i, region in enumerate(regions):
        ax = axes[i]
        sub_df = df[df['region'] == region]
        
        for exp in experiments:
            exp_df = sub_df[sub_df["experiment"] == exp]
            ax.plot(exp_df["Year"], exp_df["NBP"].cumsum(), label=exp)
            ax.set_xlim([1700, LATEST_TRENDY_YEAR])
            ax.set_title(region)
        
        if i % 2 == 0:
            ax.set_ylabel("Cumulative NBP (PgC)")
        if i >= 2:
            ax.set_xlabel("Year")
        if i == 0:
            ax.legend()
    
    plt.tight_layout()


def get_nbp_drift(df: pd.DataFrame) -> float:
    """Calculate TRENDY-defined drift in NBP over time.
    
    Computes the slope of the linear regression between Year and NBP,
    multiplied by 100 to express the trend as PgC/yr per century.

    Args:
        df (pd.DataFrame): input global NBP dataframe

    Returns:
        float: slopex100 (i.e. drift, according to TRENDY)
    """

    X = df[["Year"]]
    y = df["NBP"]

    model = LinearRegression()
    model.fit(X, y)

    drift = model.coef_[0] * 100.0

    return drift

def check_trendy_spinup_criteria(s0_df: pd.DataFrame, drift_tol: float=0.05, offset_tol: float=0.10):
    """Check if S0 simulation meets TRENDY equilibrium criteria

    Args:
        s0_df (pd.DataFrame): global NBP data for S0 with 'Year' and 'NBP' columns
        drift_tol (float, optional): tolerance for drift (PgC/yr/century). Defaults to 0.05.
        offset_tol (float, optional): tolerance for offset (PgC/yr). Defaults to 0.10.
    """
    nbp_drift = get_nbp_drift(s0_df)
    nbp_offset = abs(s0_df["NBP"].mean())
    
    passes = nbp_drift < drift_tol and nbp_offset < offset_tol
    
    return passes, nbp_drift, nbp_offset


def check_trendy_criteria(global_df: pd.DataFrame, drift_tol: float=0.05, offset_tol: float=0.10):
    """Prints a report of TRENDY equilibrium and flux criteria

    Args:
        global_df (pd.DataFrame): global NBP data with 'experiment', 'Year', and 'NBP' columns.
        drift_tol (float, optional): tolerance for drift (PgC/yr/century). Defaults to 0.05.
        offset_tol (float, optional): tolerance for offset (PgC/yr). Defaults to 0.10.
    """

    s0_df = global_df[global_df.experiment == "S0"]
    s2_df = global_df[global_df.experiment == "S2"]
    s3_df = global_df[global_df.experiment == "S3"]
    
    print("\n-------------------------------------------------------------\n")
    print("TRENDY Equilibrium Check for 'S0':\n")
    passes, nbp_drift, nbp_offset = check_trendy_spinup_criteria(s0_df, drift_tol, offset_tol)
    if passes:
        print("S0 simulation passes TRENDY equilibrium criteria.")
    else:
        print("S0 simulation DOES NOT pass TRENDY equilibrium criteria.")

    print(f"\n→ Drift:  {nbp_drift:.4f} PgC/yr/century")
    print(f"→ Offset: {nbp_offset:.4f} PgC/yr")
    
    print("\n-------------------------------------------------------------\n")
    print("TRENDY net annual land flux check from 'S3':\n")
    sink_results = check_land_flux_sink(s3_df)
    for decade, is_sink in sink_results.items():
        status = "PASS" if is_sink else "FAIL"
        print(f"  {decade}: Net land flux is a sink? {is_sink} [{status}]")
    
    print("\n-------------------------------------------------------------\n")
    print("TRENDY net annual land use flux for 1990s:")
    is_source = check_land_use_flux_source(s2_df, s3_df)
    status = "PASS" if is_source else "FAIL"
    print(f"  Land use flux is a carbon source? {is_source} [{status}]")


def check_land_flux_sink(s3_df: pd.DataFrame, start_years: list[int]=[1990, 2000, 2010]) -> dict:
    """ Checks if global net annual land flux is a carbon sink (i.e. negative flux)
    averaged over 1990s, 2000s, and 2010s decades from S3 run.

    Args:
        s3_df (pd.DataFrame): DataFrame for S3 experiment with columns ['Year', 'NBP']
        start_years (list[int], optional): _description_. Defaults to [1990, 2000, 2010].
        end_year (int, optional): last year in the range. Defaults to 2019.

    Returns:
        dict: {decade_str: bool} whether net land flux is sink (True if negative)
    """
    results = {}
    s3_df = s3_df.set_index("Year")
    for start in start_years:
        end_year = start + 9
        decade = f"{start}s"
        df_decade = s3_df.loc[start:end_year]
        net_land_flux = df_decade['NBP']
        avg_flux = net_land_flux.mean()
        results[decade] = avg_flux > 0.0 # true if sink
    return results

def check_land_use_flux_source(s2_df: pd.DataFrame, s3_df: pd.DataFrame, start_year: int=1990,
                               end_year: int=1999) -> bool:
    """Checks if global net annual land use flux (Eluc from S3 - S2) is a carbon source (positive)

    Args:
        s2_df (pd.DataFrame): DataFrame for S2 experiment with columns ['Year', 'NBP']
        s3_df (pd.DataFrame): DataFrame for S3 experiment with columns ['Year', 'NBP']
        start_year (int, optional): _description_. Defaults to 1990.
        end_year (int, optional): _description_. Defaults to 1999.

    Returns:
        bool: True if difference (S3 NBP - S2 NBP) is positive (carbon source)
    """
    
    # align years
    s2 = s2_df.set_index("Year").loc[start_year:end_year]
    s3 = s3_df.set_index("Year").loc[start_year:end_year]
    
    # calculate difference
    diff = s3["NBP"] - s2["NBP"]
    avg_diff = diff.mean()
    return avg_diff < 0.0

def calculate_global_mean(da: xr.DataArray) -> xr.DataArray:
    """Calculates global mean for a variable

    Args:
        da (xr.DataArray): input DataArray

    Returns:
        xr.DataArray: output DataArray
    """

    global_mean = da.mean(dim=["lat", "lon"])

    return global_mean


def get_data_dict(row: pd.Series) -> dict:
    """Creates a dictionary with TRENDY-relevant information about a TRENDY output variable.

    Args:
        row (pd.Series): A single row of the var_df DataFrame.

    Returns:
        dict: Dictionary with TRENDY-relevant information about the output variable.
    """
    data_dict = {
        "vars_in": row["CLM_varname"].split("-"),
        "long_name": row["long_name"],
        "output_units": row["units"],
        "fill_val": row["NA_val"],
        "conversion_factor": row["conversion_factor"],
        "global_cf": row["global_cf"],
        "frequency": row["frequency"],
        "dims": row["CMOR_dims"],
        "dimension": row["dimension"],
        "output_function": row["output_function"],
    }

    if data_dict["dimension"] == "pft":
        data_dict["stream"] = PFT_STREAM
    elif data_dict["frequency"] == "daily":
        data_dict["stream"] = DAILY_STREAM
    else:
        data_dict["stream"] = GLOBAL_STREAM

    return data_dict


def get_var_files(tseries_dir: str, stream: str, var: str) -> list[str]:
    """Returns a list of files for a variable and stream

    Args:
        tseries_dir (str): path to single-variable time-series folder
        stream (str): stream [h0, h1, h2]
        var (str): variable

    Returns:
        list[str]: list of files

    Raises:
        FileNotFoundError: can't find any files
    """
    pattern = os.path.join(tseries_dir, f"*{stream}a.{var}.*")
    var_files = sorted(glob.glob(pattern))

    if not var_files:
        raise FileNotFoundError(
            f"No files found for variable '{var}' in stream '{stream}' at '{tseries_dir}'"
        )

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


def read_in_ds(
    files: list[str],
    in_vars: list[str],
    stream: str,
    min_year: int = 1700,
    max_year: int = LATEST_TRENDY_YEAR,
) -> xr.Dataset:
    """Reads in and concatenates a dataset from a list of files, also fixes the
        time variable to fix the year 0 issue in CLM

    Args:
        files (list[str]): list of netcdf files
        in_vars (list[str]): list of variables to read in
        stream (str): file stream [h0, h1, h2]
        min_year (int, optional): minimum year to subset dataset to. Defaults to 1700.
        max_year (int, optional): maximum year to subset dataset to. 
            Defaults to constant LATEST_TRENDY_YEAR

    Returns:
        xr.Dataset: output dataset
    """

    with dask.config.set(**{"array.slicing.split_large_chunks": False}):
        ds = xr.open_mfdataset(
            files,
            combine="nested",
            concat_dim="time",
            parallel=True,
            autoclose=True,
            preprocess=functools.partial(preprocess, data_vars=in_vars),
            chunks={"time": 10},
        )

    if stream == PFT_STREAM:
        # add in these variables for PFT stream
        metadata_vars = [
            "lat",
            "lon",
            "pfts1d_itype_veg",
            "pfts1d_ixy",
            "pfts1d_jxy",
            "landfrac",
        ]
        with xr.open_dataset(files[0]) as tmp:
            for var in metadata_vars:
                if var in tmp:
                    ds[var] = tmp[var]
                else:
                    print(f"Warning: '{var}' not found in metadata file {files[0]}")

    # subset to min/max time
    ds = ds.sel(time=slice(f"{min_year}-01-01", f"{max_year}-12-31"))

    return ds


def get_base_da(ds: xr.Dataset, pft: int) -> xr.DataArray:
    """Creates a base xarray DataArray filled with NaNs, matching the lat/lon/time
    grid of the input dataset and a single PFT index.

    Args:
        ds (xr.Dataset): input dataset
        pft (int): pft index we are regridding

    Returns:
        xr.DataArray: A 4D DataArray (lat, lon, PFT, time) filled with NaNs.
    """
    shape = (len(ds.lat), len(ds.lon), 1, len(ds.time))
    coords = {"lat": ds.lat, "lon": ds.lon, "PFT": [pft], "time": ds.time}

    base_da = xr.DataArray(
        data=np.full(shape, np.nan), dims=["lat", "lon", "PFT", "time"], coords=coords
    )

    return base_da


def create_regridder(
    ds: xr.Dataset, nlon: int, pft: int
) -> tuple[xr.DataArray, xr.DataArray]:
    """Creates regridder DataArrays for regridding an input Dataset to a specific pft

    Args:
        ds (xr.Dataset): input dataset with pft mapping fields.
        nlon (int): number of longitudes in the destination grid.
        pft (int): PFT index to regrid

    Returns:
        tuple:
            - xr.DataArray: 1D index array for placing PFT values into lat/lon grid
            - xr.DataArray: boolean mask selecting the desired PFT entries
    """
    # mask selecting entries of given PFT type
    ixpft = ds.pfts1d_itype_veg == pft

    # extract x/y grid indices for selected PFT and convert to 1D flat index
    ixy = ds.pfts1d_ixy.isel(pft=ixpft) - 1
    jxy = ds.pfts1d_jxy.isel(pft=ixpft) - 1
    regridder = (jxy * nlon + ixy).astype(int).compute()

    return regridder, ixpft


def regrid_pft_ds(
    ds: xr.Dataset, pft: int, var: str, pft_names: xr.DataArray
) -> xr.Dataset:
    """Regrids a CLM history variable from a PFT-resolved dataset to a specific PFT and
    maps it onto a spatial grid.

    Args:
        ds (xr.Dataset): input Dataset
        pft (int): pft index to regrid to
        var (str): CLM history variable to regrid
        pft_names (xr.DataArray): DataArray of pft names (indexed by PFT index)

    Returns:
        xr.Dataset: output Dataset
    """

    # get base DataArray to define spatial grid
    base_grid = get_base_da(ds, pft)

    # generate regrid mask and corresponding PFT index
    regrid_mask, ixpft = create_regridder(ds, len(base_grid.lon), pft)

    # stack lat/lon into gridcell dimension
    var_out = base_grid.copy(deep=True).stack({"gridcell": ["lat", "lon"]})
    var_out[0, :, regrid_mask] = ds[var].isel(pft=ixpft)
    var_out = var_out.unstack()
    var_out.attrs = ds[var].attrs  # preserve metadata

    # apply land fraction weighting if needed
    if var == "pfts1d_wtgcell":
        var_out *= ds.landfrac

    # construct final dataset
    out_ds = xr.Dataset(
        {
            var: var_out,
            "time": ds.time,
            "pftname": xr.DataArray([pft_names[pft].values], dims="PFT"),
        }
    )

    return out_ds


def read_in_all_vars(
    vars_in: list[str],
    tseries_dir: str,
    stream: str,
    pft_names: xr.DataArray = None,
    pft: int = None,
) -> xr.Dataset:
    """Read in and merges CLM history variables in input list
        If the variable is an h1 stream, the data is regridded to the input pft

    Args:
        vars_in (list[str]): list of CLM history variables
        tseries_dir (str): path to single-variable time-series folder
        stream (str): file stream [h0, h1, 2]
        pft_names (xr.DataArray, optional): DataArray of pft names (indexed by PFT index).
            Defaults to None
        pft (int, optional): PFT index to regrid to if an h1 stream. Defaults to None.

    Raises:
        ValueError: Trying to read in an h1 stream but no pft index was supplied
        ValueError: Can't find files for variable
        ValueError: No valid datsaets read in

    Returns:
        xr.Dataset: output Dataset
    """

    ds_list = []

    for var in vars_in:
        var_stream = GLOBAL_STREAM if var in ["FLDS", "FSDS"] else stream
        files = get_var_files(tseries_dir, var_stream, var)
        ds_var = read_in_ds(files, [var], var_stream)

        if var_stream == PFT_STREAM:
            if pft is None:
                raise ValueError(f"Must supply a pft for {PFT_STREAM} streams.")

            ds_var = regrid_pft_ds(ds_var, pft, var, pft_names)

        ds_list.append(ds_var)

    if len(ds_list) == 0:
        raise ValueError("No valid datasets were read in.")

    return ds_list[0] if len(ds_list) == 1 else xr.merge(ds_list)


def sum_vars(
    ds: xr.Dataset, vars_in: list[str], var_out: str
) -> tuple[xr.Dataset, str]:
    """Sums a list of variables in an xarray Dataset and stores the result as a new variable

    Args:
        ds (xr.Dataset): input Dataset
        vars_in (list[str]): list of variables to sum
        var_out (str): name of new DataArray

    Returns:
        tuple:
         - xr.Dataset: output Dataset
         - str: string explaining how this variable was calculated
    Raises:
        ValueError: vars_in list empty
        KeyError: variable missing from dataset
    """

    if len(vars_in) == 0:
        raise ValueError("vars_in list is empty.")

    if any(var not in ds for var in vars_in):
        missing = [var for var in vars_in if var not in ds]
        raise KeyError(
            f"The following variables are missing from the dataset: {missing}"
        )

    # create an empty data array
    ds[var_out] = xr.full_like(ds[vars_in[0]], 0.0)

    # sum up input variables
    for var in vars_in:
        ds[var_out] += ds[var]

    # create string to explain how this variable was calculated
    clm_orig_var_string = " + ".join(vars_in)

    return ds, clm_orig_var_string


def find_cpool_var(vars_in: list[str], cpool_string: str) -> str:
    """Finds the matching CLM history variable for the correct carbon pool

    Args:
        vars_in (list[str]): input list of history variables
        cpool_string (str): carbon pool string to match (i.e. ACT, PAS, SLO)

    Returns:
        str: CLM history variable name
    Raises:
        ValueError: if not matching variable is found
    """
    for var in vars_in:
        if cpool_string in var:
            return var

    raise ValueError(f"No variable found matching carbon pool string: '{cpool_string}'")


def reorg_by_cpool(
    ds: xr.Dataset, var_out: str, active_var: str, slow_var: str, passive_var: str
) -> xr.Dataset:
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
    pools = []
    pool_names = ["active", "slow", "passive"]
    for var in [active_var, slow_var, passive_var]:
        pools.append(ds[var].to_dataset(name=var_out))

    ds_pools = xr.concat(pools, dim="Pool", data_vars="all")
    ds_pools = ds_pools.assign_coords(Pool=("Pool", pool_names))
    ds_pools = ds_pools.transpose("time", "Pool", "lat", "lon")

    return ds_pools


def carbon_pool(
    ds: xr.Dataset, vars_in: list[str], var_out: str
) -> tuple[xr.Dataset, str]:
    """Calculates a carbon pool variable from individual pool components (active, slow, passive).

    Args:
        ds (xr.Dataset): input Dataset
        vars_in (list[str]): List of CLM variable names expected to include ACT/SLO/PAS pools
        var_out (str): output variable name

    Returns:
        tuple:
            - xr.Dataset: output Dataset
            - str: string explaining how this variable was calculated
    Raises:
        ValueError: vars_in is empty
        ValueError: can't find cpool variable
    """
    # validate input list
    if len(vars_in) == 0:
        raise ValueError("vars_in must not be empty")

    # find matching strings
    active_var = find_cpool_var(vars_in, "ACT")
    slow_var = find_cpool_var(vars_in, "SLO")
    passive_var = find_cpool_var(vars_in, "PAS")

    # reorganize by carbon pool
    ds = reorg_by_cpool(ds, var_out, active_var, slow_var, passive_var)

    clm_orig_var_string = ", ".join(vars_in)

    return ds, clm_orig_var_string


def rh_pool(ds: xr.Dataset, vars_in: list[str], var_out: str) -> tuple[xr.Dataset, str]:
    """Calculates TRENDY's rhpool variable (Carbon Flux from individual soil pools)

    Args:
        ds (xr.Dataset): input Dataset
        vars_in (str[list]): int CLM variables needed for this output
        var_out (str): output variable name

    Returns:
        tuple[xr.Dataset, str]
          - xr.Dataset: output Dataset
          - str: string explaining how this variable was calculated
    """
    # define intermediate flux names
    flux_from_act = "FROM_ACT"
    flux_from_slo = "FROM_SLO"
    flux_from_pas = "FROM_PAS"

    # compute total flux from each pool
    ds[flux_from_act] = ds["SOM_ACT_C_TO_SOM_PAS_C"] + ds["SOM_ACT_C_TO_SOM_SLO_C"]
    ds[flux_from_slo] = ds["SOM_SLO_C_TO_SOM_ACT_C"] + ds["SOM_SLO_C_TO_SOM_PAS_C"]
    ds[flux_from_pas] = ds["SOM_PAS_C_TO_SOM_ACT_C"]

    # reorganize into 'Pool' dimension
    ds_pools = reorg_by_cpool(ds, var_out, flux_from_act, flux_from_slo, flux_from_pas)

    clm_orig_var_string = (
        "active: SOM_ACT_C_TO_SOM_PAS_C, SOM_ACT_C_TO_SOM_SLO_C; "
        "slow: SOM_SLO_C_TO_SOM_ACT_C, SOM_SLO_C_TO_SOM_PAS_C; "
        "passive: SOM_PAS_C_TO_SOM_ACT_C"
    )

    return ds_pools, clm_orig_var_string


def black_sky_albedo(
    ds: xr.Dataset, vars_in: list[str], var_out: str
) -> tuple[xr.Dataset, str]:
    """Calculates black sky albedo

    Args:
        ds (xr.Dataset): input Dataset
        vars_in (str[list]): Required CLM input variables: ['FSRND', 'FSRVD', 'FSDSND', 'FSDSVD']
        var_out (str): output variable name

    Returns:
        tuple:
         - xr.Dataset: output Dataset
         - str: string explaining how this variable was calculated
    Raises:
        ValueError: missing variable in dataset
    """
    required_vars = ["FSRND", "FSRVD", "FSDSND", "FSDSVD"]
    missing = [v for v in required_vars if v not in ds]
    if missing:
        raise ValueError(f"Missing required variables in dataset: {missing}")

    # calculate black sky albedo
    numerator = ds["FSRND"] + ds["FSRVD"]
    denominator = ds["FSDSND"] + ds["FSDSVD"]
    bs_alb = numerator / denominator

    # convert to dataset
    data_out = bs_alb.to_dataset(name=var_out)

    # preserve PFT dimension if it exists
    if "pftname" in ds.coords or "pftname" in ds:
        data_out["pftname"] = ds.pftname

    clm_orig_var_string = "(FSRND + FSRVD)/(FSDSND + FSDSVD)"

    return data_out, clm_orig_var_string


def white_sky_albedo(
    ds: xr.Dataset, vars_in: list[str], var_out: str
) -> tuple[xr.Dataset, str]:
    """Calculates white sky albedo

    Args:
        ds (xr.Dataset): input Dataset
        vars_in (str[list]): int CLM variables needed for this output
        var_out (str): output variable name

    Returns:
        tuple:
         - xr.Dataset: output Dataset
         - str: string explaining how this variable was calculated
    Raises:
        VaueError: missing variable
    """
    required_vars = ["FSRNI", "FSRVI", "FSDSNI", "FSDSVI"]
    missing = [v for v in required_vars if v not in ds]
    if missing:
        raise ValueError(f"Missing required variables in dataset: {missing}")

    # calculate white sky albedo
    numerator = ds["FSRNI"] + ds["FSRVI"]
    denominator = ds["FSDSNI"] + ds["FSDSVI"]
    ws_alb = numerator / denominator

    # convert to dataset
    data_out = ws_alb.to_dataset(name=var_out)

    # preserve PFT dimension if it exists
    if "pftname" in ds.coords or "pftname" in ds:
        data_out["pftname"] = ds.pftname

    clm_orig_var_string = "(FSRNI + FSRVI)/(FSDSNI + FSDSVI)"

    return data_out, clm_orig_var_string

def net_radiation(
    ds: xr.Dataset, vars_in: list[str], var_out: str
) -> tuple[xr.Dataset, str]:
    """Calculates net radiation

    Args:
        ds (xr.Dataset): input Dataset
        vars_in (str[list]): int CLM variables needed for this output
        var_out (str): output variable name

    Returns:
        tuple:
         - xr.Dataset: output Dataset
         - str: string explaining how this variable was calculated
    Raises:
        VaueError: missing variable
    """
    required_vars = ["FLDS", "FIRE", "FSDS", "FSR"]
    missing = [v for v in required_vars if v not in ds]
    if missing:
        raise ValueError(f"Missing required variables in dataset: {missing}")

    # calculate net radiation
    rn = ds.FLDS - ds.FIRE + ds.FSDS - ds.FSR
    
    # convert to dataset
    data_out = rn.to_dataset(name=var_out)

    # preserve PFT dimension if it exists
    if "pftname" in ds.coords or "pftname" in ds:
        data_out["pftname"] = ds.pftname

    clm_orig_var_string = "FLDS - FIRE + FSDS - FSR"

    return data_out, clm_orig_var_string


def calculate_annual_mean(da: xr.DataArray) -> xr.DataArray:
    """Calculates the annual mean for a DataArray

    Args:
        da (xr.DataArray): input DataArray

    Returns:
        xr.DataArray: output DataArray
    """

    months = da["time.daysinmonth"]
    annual_mean = 1 / 365 * (months * da).groupby("time.year").sum()
    annual_mean.name = da.name
    annual_mean.attrs = da.attrs

    return annual_mean


def make_annual(ds: xr.Dataset, var_out: str) -> xr.Dataset:
    """Calculates an annual dataset

    Args:
        ds (xr.DataSet): input Dataset
        var_out (str): output variable name

    Returns:
        xr.Dataset: output Dataset
    """

    # calculate annual mean
    annual_mean = calculate_annual_mean(ds[var_out])
    data_out = annual_mean.to_dataset(name=var_out)

    # rename year to time as per TRENDY instructions
    data_out = data_out.rename({"year": "time"})
    data_out["time"].attrs = {"long_name": "year", "units": "yr"}

    return data_out


def create_trendy_var(
    file_name: str,
    data_dict: dict,
    func_dict: dict,
    var_out: str,
    tseries_dir: str,
    clobber: bool = False,
    pft_names: xr.DataArray = None,
    pft: int = None,
):
    """Creates and writes out a TRENDY variable based on information in the data_dict and func_dict

    Args:
        file_name (str): full path of file name to write to
        data_dict (dict): dictionary with TRENDY-relevant information about a TRENDY output variable
        func_dict (dict): dictionary mapping to functions used to calculate output  variables
        var_out (str): output variable name
        tseries_dir (str): path to single-variable time series data
        clobber (bool, optional): whether or not to overwrite files. Defaults to False.
        pft_names (xr.DataArray, optional): DataArray of pft names (indexed by PFT index). 
            Defaults to None.
        pft (int, optional): PFT index to regrid to for h1 streams. Defaults to None.

    Raises:
        KeyError: Can't find function in dictionary
        ValueError: Unknown frequency for variable
        ValueError: Unknown dimension for variable
    """

    # check if file exists and whether we want to overwrite
    if os.path.exists(file_name):
        if not clobber:
            print(
                f"Skipping {var_out}{f' for PFT {pft}' if pft is not None else ''} (exists)"
            )
            return
        os.remove(file_name)

    # make directories if they don't exist
    os.makedirs(os.path.dirname(file_name), exist_ok=True)

    vars_in = data_dict["vars_in"]
    stream = data_dict["stream"]

    # read in all variables required for this output variable
    # if an h1 stream, it is also regridded to the input pft
    ds = read_in_all_vars(vars_in, tseries_dir, stream, pft_names, pft)

    # store original variable attributes
    units = {var: ds[var].attrs.get("units", "unknown") for var in vars_in}
    long_names = {var: ds[var].attrs.get("long_name", "unknown") for var in vars_in}

    # create output variable using appropriate function
    try:
        ds, func_string = func_dict[data_dict["output_function"]](ds, vars_in, var_out)
    except KeyError:
        print(
            f"Unknown function '{data_dict['output_function']}' for variable {var_out}"
        )
        raise

    # unit conversion
    if var_out == "burntArea":
        fractional = ds["time.daysinmonth"] * ds[var_out] * 24 * 60 * 60
        ds[var_out] = fractional
    else:
        ds[var_out] *= data_dict["conversion_factor"]

    # temporal aggregation
    if data_dict["frequency"] == "annual":
        data_out = make_annual(ds, var_out)
    elif data_dict["frequency"] == "monthly":
        data_out = ds[var_out].to_dataset(name=var_out)
    elif data_dict["frequency"] == "daily":
        data_out = ds[var_out].to_dataset(name=var_out)
    else:
        raise ValueError(
            f"Unknown frequency '{data_dict['frequency']}' for variable {var_out}"
        )

    # rename dimensions if needed
    if data_dict["dimension"] == "soil_layer":
        rename_map = DIM_RENAME_MAP.get(var_out)
        if rename_map is None:
            raise ValueError(f"No dimension rename rule for variable '{var_out}'")
        data_out = data_out.rename(rename_map)

    # set variable attributes
    attrs = {
        "long_name": data_dict["long_name"],
        "units": data_dict["output_units"],
        "_FillValue": data_dict["fill_val"],
        "CLM-TRENDY_unit_conversion_factor": (
            "days_in_month*24*60*60"
            if var_out == "burntArea"
            else data_dict["conversion_factor"]
        ),
        "CLM_orig_var_name": func_string,
    }
    for var in vars_in:
        data_out[var_out].attrs[f"CLM_orig_attr_{var}_units"] = units[var]
        data_out[var_out].attrs[f"CLM_orig_attr_{var}_long_name"] = long_names[var]

    data_out[var_out].attrs = attrs

    # add PFT dimensions and attributes if needed
    if data_dict["stream"] == PFT_STREAM:
        data_out["pftname"] = ds.pftname
        data_out["pftname"].attrs = {"long_name": "pft name"}
        data_out["PFT"].attrs = {"long_name": "pft index"}
        data_out = data_out.transpose("time", "PFT", "lat", "lon")

    # fill missing values
    data_out = data_out.fillna(data_dict["fill_val"])

    # final encoding
    encoding = {var_out: {"dtype": "float32"}}
    for key in ["lat", "lon", "time"]:
        encoding[key] = NETCDF_ENCODING[key]
    if stream == PFT_STREAM:
        encoding["PFT"] = NETCDF_ENCODING["PFT"]
        encoding["pftname"] = NETCDF_ENCODING["pftname"]

    # write output
    data_out.to_netcdf(
        file_name,
        unlimited_dims="PFT" if stream == PFT_STREAM else None,
        encoding=encoding,
    )


def create_all_trendy_vars(
    out_dir: str,
    var_df_file: str,
    exp: str,
    top_dir: str,
    hist_dir: str,
    pft_list: list[int],
    clm_param_file: str,
    clobber: bool = False,
):
    """Create all TRENDY variables

    Args:
        out_dir (str): path to output directory
        var_df_file (str): file name describing TRENDY variables to create
        exp (str): TRENDY experiment [S0, S1,...]
        top_dir (str): top directory where single-variable time series data are located
        hist_dir (str): path to history directory for this TRENDY experiment
        pft_list (list[int]): list of PFTs for this simulation
        clm_param_file (str): path to CLM parameter file
        clobber (bool, optional): whether or not to overwrite data. Defaults to False.
    """
    # load variable definitions
    try:
        var_df = pd.read_csv(var_df_file)
    except FileNotFoundError as e:
        raise RuntimeError(f"Variable definition file not found: {var_df_file}") from e
    
    # filter on variables we are computing
    var_df = var_df[var_df['CLM_run'] == 1]

    # load CLM parameter dataset
    try:
        clm_param = xr.open_dataset(clm_param_file)
    except FileNotFoundError as e:
        raise RuntimeError(f"CLM parameter file not found: {clm_param_file}") from e

    # decode PFT names
    pft_names = xr.DataArray(
        [
            pft.decode("utf-8").strip() if isinstance(pft, bytes) else str(pft).strip()
            for pft in clm_param.pftname.values
        ],
        dims="PFT",
    )

    # dictionary for calculating output variables
    funcs = {
        "sum": sum_vars,
        "carbon_pool": carbon_pool,
        "rh_pool": rh_pool,
        "black_sky_albedo": black_sky_albedo,
        "white_sky_albedo": white_sky_albedo,
        "rn": net_radiation,
    }

    # loop through all output variables
    for _, row in var_df.iterrows():

        # handle landCoverFrac separately
        if row["output_varname"] == 'landCoverFrac':
            regrid_all_landcoverfracs(
                hist_dir=hist_dir,
                exp=exp,
                var_row=row,
                var_out=row['output_varname'],
                pft_list=pft_list,
                out_dir=out_dir,
                pft_names=pft_names,
                clobber=clobber
            )
            continue
        
        # handle oceanCoverFrac separately
        if row["output_varname"] == 'oceanCoverFrac':
            create_oceanfrac(
                hist_dir=hist_dir,
                exp=exp,
                var_row=row,
                var_out=row['output_varname'],
                out_dir=out_dir,
                clobber=clobber
            )
            continue

        # grab data dictionary for this output variable
        data_dict = get_data_dict(row)
        if data_dict['frequency'] == 'daily':
            tseries_dir = os.path.join(top_dir, f'{exp}/day_1')
        else:
            tseries_dir = os.path.join(top_dir, f'{exp}/month_1')

        if data_dict["stream"] == PFT_STREAM:
            if not pft_list:
                raise ValueError(
                    (
                        f"PFT list is required for variable {row['output_varname']} "
                        f"with stream {PFT_STREAM}"
                    )

                )

            for pft in pft_list:
                file_name = os.path.join(
                    out_dir, exp, f"CLM6.0_{exp}_{row['output_varname']}_PFT{pft:02}.nc"
                )

                create_trendy_var(
                    file_name,
                    data_dict,
                    funcs,
                    row["output_varname"],
                    tseries_dir,
                    clobber=clobber,
                    pft_names=pft_names,
                    pft=pft,
                )
        else:
            file_name = os.path.join(
                out_dir, exp, f"CLM6.0_{exp}_{row['output_varname']}.nc"
            )
            create_trendy_var(
                file_name,
                data_dict,
                funcs,
                row["output_varname"],
                tseries_dir,
                clobber=clobber,
            )
            
    # create the grid area output file
    create_gridarea(hist_dir, exp, 'gridArea', out_dir, clobber=clobber)


def regrid_all_landcoverfracs(
    hist_dir: str,
    exp: str,
    var_row: pd.Series,
    var_out: str,
    pft_list: list[int],
    pft_names: xr.DataArray,
    out_dir: str,
    clobber: bool=False,
):
    """Creates and writes out the TRENDY landCoverFrac variable for all PFTs

    Args:
        hist_dir (str): path to history directory for this TRENDY experiment
        exp (str): TRENDY experiment name [S0, S1,...]
        var_row (pd.Series): row of the DataFrame describing this TRENDY variable
        var_out (str): output variable name
        pft_list (list[int]): list of pfts to convert
        pft_names (xr.DataArray): decoded PFT names
        out_dir (str): output directory
        clobber (bool, optional): whether or not to overrite files. Defaults to False.
    """

    # grab data dictionary for this output variable
    data_dict = get_data_dict(var_row)
    var_in = data_dict["vars_in"][0]

    # get land cover fraction for all pfts
    ds = read_landcoverfrac_ds(hist_dir, exp, var_in)

    # regrid each pft
    for pft in pft_list:

        file_name = os.path.join(
            out_dir, exp, f"CLM6.0_{exp}_{var_out}_PFT{pft:02}.nc"
        )
        
        # check if file exists and whether we want to overwrite
        if os.path.exists(file_name):
            if not clobber:
                print(f"Skipping {var_out} for PFT {pft} (exists)")
                continue
            os.remove(file_name)
        
        # regrid the landcover frac
        data_out = regrid_landcoverfrac(ds, var_out, var_in, data_dict, pft, pft_names)
        
        # final encoding
        encoding = {var_out: {"dtype": "float32"}}
        for key in ["lat", "lon", "time", "PFT", "pftname"]:
            encoding[key] = NETCDF_ENCODING[key]
        
        # write output
        data_out.to_netcdf(file_name, unlimited_dims="PFT", encoding=encoding)

def create_oceanfrac(hist_dir: str, exp: str, var_row: pd.Series, var_out: str, 
                     out_dir: str, clobber: bool=False):
    """Creates and writes out the TRENDY oceanCoverFrac variable

    Args:
        hist_dir (str): path to history directory for this TRENDY experiment
        exp (str): TRENDY experiment name [S0, S1,...]
        var_row (pd.Series): row of the DataFrame describing this TRENDY variable
        var_out (str): output variable name
        out_dir (str): output directory
        clobber (bool, optional): Whether or not to overwrite files. Defaults to False.
    """
    
    # grab data dictionary for this output variable
    data_dict = get_data_dict(var_row)

    file_name = os.path.join(out_dir, exp, f"CLM6.0_{exp}_{var_out}.nc")
    
    if os.path.exists(file_name):
        if not clobber:  
            print(f"Skipping {var_out} (exists)")
            return
        os.remove(file_name)

    # get land area and conver to ocean fraction
    ds = get_land_area(hist_dir)
    ds[var_out] = 1.0 - ds.landfrac
    data_out = ds[var_out].to_dataset(name=var_out)

    # set attributes
    data_out[var_out].attrs = {
        "long_name": data_dict["long_name"],
        "units": "",
        "_FillValue": data_dict["fill_val"],
        "CLM-TRENDY_unit_conversion_factor": data_dict["conversion_factor"],
        "CLM_orig_var_name": "1.0 - landfrac",
    }
    data_out[var_out].attrs["CLM_orig_attr_landfrac_long_name"] = ds.landfrac.attrs[
        "long_name"
    ]
    data_out = data_out.fillna(data_dict["fill_val"])

    # final encoding    
    encoding = {var_out: {"dtype": "float32"}}
    for key in ["lat", "lon"]:
        encoding[key] = NETCDF_ENCODING[key]
    
    # write output
    data_out.to_netcdf(file_name, encoding=encoding)


def create_gridarea(hist_dir: str, exp: str, var_out: str, out_dir: str, clobber=False):
    """Creates the TRENDY output variable gridArea

    Args:
        hist_dir (str): path to history directory for this TRENDY experiment
        exp (str): TRENDY experiment name [S0, S1,...]
        var_out (str): output variable name
        out_dir (str): output directory
        clobber (bool, optional): Whether or not to overwrite files. Defaults to False.
    """

    file_name = os.path.join(out_dir, exp, f"CLM6.0_{exp}_{var_out}.nc")

    if os.path.exists(file_name):
        if not clobber:  
            print(f"Skipping {var_out} (exists)")
            return
        os.remove(file_name)
    
    # get land area
    ds = get_land_area(hist_dir)
    ds[var_out] = ds.area
    data_out = ds[var_out].to_dataset(name=var_out)

    # set attributes
    data_out[var_out].attrs = {
        "long_name": "gridcell area",
        "units": "m2",
        "_FillValue": -9999.0,
        "CLM_orig_var_name": "area",
        "CLM-TRENDY_unit_conversion_factor": 1e6,
    }
    data_out[var_out].attrs["CLM_orig_attr_area_units"] = "km2"
    data_out[var_out].attrs["CLM_orig_attr_area_long_name"] = ds.area.attrs["long_name"]
    data_out = data_out.fillna(-9999.0)

    # final encoding
    encoding = {var_out: {"dtype": "float32"}}
    for key in ["lat", "lon"]:
        encoding[key] = NETCDF_ENCODING[key]

    # write output
    data_out.to_netcdf(file_name, encoding=encoding)
