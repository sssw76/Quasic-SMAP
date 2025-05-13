import os
from tqdm import tqdm
import warnings
import glob
from osgeo import gdal
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy import stats
from scipy.optimize import fsolve

warnings.filterwarnings('ignore')


def save_time_series(data_dir, output_file):
    """Read and save time series data to npy file"""
    tif_files = sorted(glob.glob(os.path.join(data_dir, "*.tif")))
    if not tif_files:
        return None

    img1 = gdal.Open(tif_files[0])
    height = img1.RasterYSize
    width = img1.RasterXSize
    img1 = None

    pixel_time_series = np.empty((height, width, len(tif_files)), dtype=np.float32)
    dates = []

    print(f"\nReading and saving time series data in {data_dir} ...")
    for i, tif_file in enumerate(tqdm(tif_files)):
        try:
            date_str = os.path.basename(tif_file).split('.')[0][0:8]
            dates.append(date_str)

            img = gdal.Open(tif_file)
            if img is not None:
                data = np.array(img.GetRasterBand(1).ReadAsArray())
                data = preprocess_data(data)
                pixel_time_series[:, :, i] = data
                img = None
                del data
        except Exception as e:
            print(f"Error loading {tif_file}: {str(e)}")
            pixel_time_series[:, :, i] = np.nan

    # Save data and dates
    np.save(output_file, pixel_time_series)
    np.save(output_file.replace('.npy', '_dates.npy'), dates)
    return dates

def load_time_series(npy_file):
    """Load time series data"""
    pixel_time_series = np.load(npy_file)
    dates = np.load(npy_file.replace('.npy', '_dates.npy'))
    return pixel_time_series, dates

def preprocess_data(data):
    """Preprocess data, keep only valid values in [0, 1]"""
    data = np.array(data, dtype=float)
    mask = (data < 0) | (data > 1) | np.isnan(data) | (data == -9999)
    data[mask] = np.nan
    return data


def dist_from_mean_std(mu: float, sigma: float, dist_type: str):
    """Generate distribution object from mean and std deviation"""
    if dist_type.lower() == 'normal':
        dist = stats.norm(loc=mu, scale=sigma)
        a, b = mu, sigma
    elif dist_type.lower() == 'beta':
        def f(x):
            a, b = x.tolist()
            return [
                a / (a + b) - mu,
                a * b / ((a + b + 1) * (a + b) ** 2) - sigma ** 2
            ]
        try:
            a, b = fsolve(f, np.array([1, 1]))
        except ValueError:
            print("fsolve did not converge to a solution.")
            return None, None, None
        dist = stats.beta(a=a, b=b)
    else:
        raise ValueError(f'Unsupported distribution type: {dist_type}')

    return dist, a, b


def beta_cdf_correction_pixel(valid_ref, valid_target):
    """Apply Beta CDF correction for a single pixel's valid data"""
    try:
        # Fit beta distributions
        mean_s = float(np.mean(valid_ref))
        std_s = float(np.std(valid_ref))
        max_s = float(np.max(valid_ref))
        min_s = float(np.min(valid_ref))

        mean_c = float(np.mean(valid_target))
        std_c = float(np.std(valid_target))
        max_c = float(np.max(valid_target))
        min_c = float(np.min(valid_target))

        # Fit distributions
        dist_obj_s, s_a, s_b = dist_from_mean_std(mu=mean_s, sigma=std_s, dist_type="beta")
        dist_obj_c, c_a, c_b = dist_from_mean_std(mu=mean_c, sigma=std_c, dist_type="beta")

        if dist_obj_s is None or dist_obj_c is None:
            return None, None

        # Calculate quantiles
        quantiles = [0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
        ppf_s = dist_obj_s.ppf(quantiles)
        ppf_c = dist_obj_c.ppf(quantiles)

        # Add max and min
        min_s = min(min_s, ppf_s[0])
        max_s = max(max_s, ppf_s[-1])
        min_c = min(min_c, ppf_c[0])
        max_c = max(max_c, ppf_c[-1])
        ppf_s = np.insert(ppf_s, 0, min_s)
        ppf_s = np.append(ppf_s, max_s)
        ppf_c = np.insert(ppf_c, 0, min_c)
        ppf_c = np.append(ppf_c, max_c)

        # Build transfer function
        transfer_func = interp1d(ppf_c, ppf_s,
                                 bounds_error=False,
                                 fill_value='extrapolate')

        return transfer_func, (s_a, s_b, c_a, c_b)

    except Exception as e:
        return None, None


def beta_cdf_correction(reference_data, target_data):
    """
    Apply Beta CDF correction on each pixel location.
    reference_data: reference data (height, width, time)
    target_data: target data (height, width, time)
    Return: array of transfer functions and parameter array
    """
    height, width, _ = reference_data.shape
    transfer_funcs = np.empty((height, width), dtype=object)
    params = np.empty((height, width), dtype=object)

    print("Performing pixel-wise BCDF correction...")
    for i in tqdm(range(height)):
        for j in range(width):
            # Get time series of current pixel
            ref_series = reference_data[i, j, :]
            target_series = target_data[i, j, :]

            # Get valid data
            valid_ref = ref_series[~np.isnan(ref_series)]
            valid_target = target_series[~np.isnan(target_series)]

            if len(valid_ref) >= 3 and len(valid_target) >= 3:
                transfer_func, param = beta_cdf_correction_pixel(valid_ref, valid_target)
                transfer_funcs[i, j] = transfer_func
                params[i, j] = param

    return transfer_funcs, params


def save_transfer_funcs(transfer_funcs, params, output_dir):
    """Save transfer functions and parameters to files"""
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "transfer_functions.npy"), transfer_funcs)
    np.save(os.path.join(output_dir, "beta_parameters.npy"), params)


def save_corrected_tif(data, template_file, output_file):
    """Save corrected data as a TIF file"""
    template_ds = gdal.Open(template_file)
    driver = gdal.GetDriverByName('GTiff')

    ds = driver.Create(
        output_file,
        template_ds.RasterXSize,
        template_ds.RasterYSize,
        1,
        gdal.GDT_Float32
    )

    ds.SetGeoTransform(template_ds.GetGeoTransform())
    ds.SetProjection(template_ds.GetProjection())

    ds.GetRasterBand(1).WriteArray(data)
    ds.GetRasterBand(1).SetNoDataValue(-9999)

    ds = None
    template_ds = None


def apply_correction(data, transfer_funcs):
    """Apply transfer function to data for correction"""
    height, width = data.shape
    corrected = np.full_like(data, np.nan)

    for i in range(height):
        for j in range(width):
            if transfer_funcs[i, j] is not None and not np.isnan(data[i, j]):
                try:
                    corrected[i, j] = transfer_funcs[i, j](data[i, j])
                    corrected[i, j] = np.clip(corrected[i, j], 0, 1)
                except:
                    corrected[i, j] = np.nan

    return corrected


def apply_saved_correction(input_tif, transfer_funcs_file, output_tif):
    """Apply saved transfer function to correct new TIF file"""
    # Load transfer function
    transfer_funcs = np.load(transfer_funcs_file, allow_pickle=True)

    # Read input TIF
    ds = gdal.Open(input_tif)
    data = ds.GetRasterBand(1).ReadAsArray()

    # Apply correction
    corrected = apply_correction(data, transfer_funcs)

    # Save corrected result
    save_corrected_tif(corrected, input_tif, output_tif)

    return corrected


def main():
    # Configuration
    cci_dir = r"F:\sm\CCI(8.1)\0215-22"
    smap_dir = r"F:\sm\SMAP\9km_SMAP\0309Resample\01nan"
    output_dir = r"F:\GSSM\Relationship"
    temp_dir = r"F:\GSSM\Relationship\temp"

    # Create necessary directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)

    # Set time series data file paths
    cci_npy = os.path.join(temp_dir, 'cci_series.npy')
    smap_npy = os.path.join(temp_dir, 'smap_series.npy')

    # Check if time series data files need to be generated
    if not os.path.exists(cci_npy):
        print("Generating CCI time series data file...")
        save_time_series(cci_dir, cci_npy)

    if not os.path.exists(smap_npy):
        print("Generating SMAP time series data file...")
        save_time_series(smap_dir, smap_npy)

    # Load time series data
    print("Loading time series data...")
    cci_series, cci_dates = load_time_series(cci_npy)
    smap_series, smap_dates = load_time_series(smap_npy)
    print("Time series data loaded.")

    # Pixel-level BCDF correction
    print("Starting BCDF correction...")
    transfer_funcs, params = beta_cdf_correction(smap_series, cci_series)

    # Save transfer functions and parameters
    print("Saving transfer functions and parameters...")
    save_transfer_funcs(transfer_funcs, params, output_dir)

    # Correct and save each timestep as TIF file
    print("Generating corrected TIF files...")
    for i, date in enumerate(cci_dates):
        print(f"Processing {date} ...")
        current_data = cci_series[:, :, i]
        corrected = apply_correction(current_data, transfer_funcs)

        output_file = os.path.join(output_dir, f"{date}.tif")
        template_file = glob.glob(os.path.join(cci_dir, f"{date}*.tif"))[0]
        save_corrected_tif(corrected, template_file, output_file)

    print("All processing completed!")


if __name__ == "__main__":
    main()