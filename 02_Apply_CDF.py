import os
from tqdm import tqdm
import glob
from osgeo import gdal
import numpy as np


def apply_correction_to_folders(root_dir, transfer_funcs_file, output_dir):
    """
    Apply correction to CCI images in multiple year folders

    Parameters:
    -----------
    root_dir : str
        Root directory containing multiple year folders
    transfer_funcs_file : str
        Path to saved transfer function file (transfer_functions.npy)
    output_dir : str
        Output folder path for corrected images
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Get all year folders
    year_folders = [f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))]

    # Load transfer functions
    print("Loading transfer functions...")
    transfer_funcs = np.load(transfer_funcs_file, allow_pickle=True)

    print(f"Start processing {len(year_folders)} year folders...")
    for year_folder in sorted(year_folders):
        year_path = os.path.join(root_dir, year_folder)
        print(f"\nProcessing year: {year_folder}")

        # Get all tif files for this year
        tif_files = glob.glob(os.path.join(year_path, "*.tif"))

        for tif_file in tqdm(tif_files, desc=f"Processing {year_folder}"):
            try:
                # Extract date string from file name
                date_str = os.path.basename(tif_file).split('-')[5][:8]

                # Construct output file path
                output_file = os.path.join(output_dir, f"{date_str}.tif")

                # Skip if the output file already exists
                if os.path.exists(output_file):
                    continue

                # Read input TIF
                ds = gdal.Open(tif_file)
                data = ds.GetRasterBand(1).ReadAsArray()

                # Apply correction
                corrected = apply_correction(data, transfer_funcs)

                # Save the corrected result
                save_corrected_tif(corrected, tif_file, output_file)

                ds = None

            except Exception as e:
                print(f"Error processing file {tif_file}: {str(e)}")


def apply_correction(data, transfer_funcs):
    """Apply transfer functions to data for correction"""
    height, width = data.shape
    corrected = np.full_like(data, np.nan)

    # Create valid value mask
    valid_mask = (data >= 0) & (data <= 1)

    for i in range(height):
        for j in range(width):
            # Only process valid data within range
            if (transfer_funcs[i, j] is not None and
                valid_mask[i, j] and
                not np.isnan(data[i, j])):
                try:
                    corrected[i, j] = transfer_funcs[i, j](data[i, j])
                    # Ensure output is also within 0-1 range
                    corrected[i, j] = np.clip(corrected[i, j], 0, 1)
                except:
                    corrected[i, j] = np.nan

    return corrected



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


# Example usage
if __name__ == "__main__":
    # Set paths
    root_dir = r"E:\sm\CCI(8.1)\03year"  # Root directory of CCI data, containing multiple year folders
    transfer_funcs_file = r"E:\GSSM\Relationship\transfer_functions.npy"  # Previously saved transfer function file
    output_dir = r"E:\GSSM\CorrectionResult_1978-2015"  # Output path for corrected images

    # Execute correction
    apply_correction_to_folders(root_dir, transfer_funcs_file, output_dir)