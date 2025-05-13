# -*- coding: utf-8 -*-
from Drought_Event_Class import query_tif
from osgeo import gdal
import xgboost as xgb
import numpy as np
import user_fun
import time
import os

def calculate_month(s1: str, s2: str):
    """
    :param s1: Start time string. Format: '1979-01'
    :param s2: End time string.
    :return: Month Number.
    """
    s1_month = int(s1.split('-')[1])
    s2_month = int(s2.split('-')[1])
    s1_year = int(s1.split('-')[0])
    s2_year = int(s2.split('-')[0])
    return13 - s1_month + s2_month + (s2_year - s1_year - 1) * 12


def unify_period(files_list: list, s: str, e: str):
    """
    :param files_list: Files list of factors.
    :param s: Start date in format:'YYYYMMDD'.
    :param e: As above.
    :return: Screened files list.
    """
    i = 0
    j = 0
    for i, file in enumerate(files_list):
        if os.path.basename(file).split('.')[0] == s:
            break
    for j, file in enumerate(files_list):
        if os.path.basename(file).split('.')[0] == e:
            break
    return files_list[i:j + 1]


def get_month_name(date_str):
    """
    Convert date string to month name
    :param date_str: Date string (YYYYMMDD)
    :return: Month name (Jan, Feb, etc.)
    """
    month_map = {
        '01': 'Jan', '02': 'Feb', '03': 'Mar', '04': 'Apr',
        '05': 'May', '06': 'Jun', '07': 'Jul', '08': 'Aug',
        '09': 'Sep', '10': 'Oct', '11': 'Nov', '12': 'Dec'
    }
    month = date_str[4:6]
    return month_map[month]


print("Starting prediction at:", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

# Set input paths
# Loading Factors
Ndvi_path = r"F:\GSSM\03_gap\02_wrap\01_NDVI\fill_time"
Albedo_path = r"F:\GSSM\03_gap\02_wrap\02_ALBEDO"
SurfT_path = r"F:\GSSM\03_gap\02_wrap\03_soil_temperature"
Rainf_path = r"F:\GSSM\03_gap\02_wrap\07_total_precipitation_sum"
CraSM_path = r"F:\GSSM\03_gap\02_wrap\05_volumetric_soil_water_layer"
Tair_path = r"F:\GSSM\03_gap\02_wrap\06_temperature_2m"
SCF_path = r"F:\GSSM\03_gap\02_wrap\08_Soil-Component-Fraction"
Pos_path = r"F:\GSSM\03_gap\02_wrap\09_Position"
Dem_path = r"F:\GSSM\03_gap\02_wrap\10_GTOPO30DEM"

# Set model and data paths
models_path = r"F:\GSSM\03_gap\03_model"  # Daily scale model path
CCI_path = os.path.join(r"F:\GSSM\02_BCDF\对应关系")  # Daily scale data path
out_path = r"F:\GSSM\03_gap\04_products"

# Create output directory
user_fun.makedir(out_path)

# Set time range
start = '20150331'  # Start date
end = '20150501'  # End date

# Get file lists
Ndvi_files = unify_period(user_fun.file_search(Ndvi_path, '.tif'), s=start, e=end)
Albedo_files = unify_period(user_fun.file_search(Albedo_path, '.tif'), s=start, e=end)
SurfT_files = unify_period(user_fun.file_search(SurfT_path, '.tif'), s=start, e=end)
Rainf_files = unify_period(user_fun.file_search(Rainf_path, '.tif'), s=start, e=end)
CraSM_files = unify_period(user_fun.file_search(CraSM_path, '.tif'), s=start, e=end)
Tair_files = unify_period(user_fun.file_search(Tair_path, '.tif'), s=start, e=end)
CCI_files = unify_period(user_fun.file_search(CCI_path, '.tif'), s=start, e=end)

# Read static data
SCF_files = user_fun.file_search(SCF_path, '.tif')
Pos_files = user_fun.file_search(Pos_path, '.tif')
Dem_files = user_fun.file_search(Dem_path, '.tif')

Fraction_Clay = SCF_files[0]
Fraction_Sand = SCF_files[1]
Fraction_Silt = SCF_files[2]
Pos_x = Pos_files[0]
Pos_y = Pos_files[1]
Dem = Dem_files[0]

# Get image information
im_width = query_tif(Dem)[0]
im_height = query_tif(Dem)[1]
im_projection = query_tif(Dem)[2]
im_geotransform = query_tif(Dem)[3]

# Create log file
log_file = open(os.path.join(out_path, 'prediction_log.txt'), 'w')

# Start prediction
print(f"Starting predictions from {start} to {end}")
print(f"Total files to process: {len(Ndvi_files)}")
start_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))

for day_index in range(len(Ndvi_files)):
    try:
        current_date = os.path.basename(Ndvi_files[day_index]).split('.')[0]
        month_name = get_month_name(current_date)

        # Load corresponding month model
        model_file = os.path.join(models_path, f'{month_name}-1.model')

        if not os.path.exists(model_file):
            print(f'Model file not found: {model_file}')
            continue

        # Read data
        ndvi_data = query_tif(Ndvi_files[day_index])[4]
        albedo_data = query_tif(Albedo_files[day_index])[4]
        surft_data = query_tif(SurfT_files[day_index])[4]
        rainf_data = query_tif(Rainf_files[day_index])[4]
        crasm_data = query_tif(CraSM_files[day_index])[4]
        tair_data = query_tif(Tair_files[day_index])[4]
        cci_data = query_tif(CCI_files[day_index])[4]

        dem_data = query_tif(Dem)[4]
        clay_data = query_tif(Fraction_Clay)[4]
        sand_data = query_tif(Fraction_Sand)[4]
        silt_data = query_tif(Fraction_Silt)[4]
        xpos_data = query_tif(Pos_x)[4]
        ypos_data = query_tif(Pos_y)[4]

        # Create date data (using year)
        date_data = np.full((im_height, im_width), float(current_date[:4]))

        # Find locations to predict
        loc = np.where(
            ((np.isnan(cci_data)) | (cci_data == -9999.0)) &
            (ndvi_data != -9999.0) &
            (albedo_data != -9999.0) &
            (surft_data != -9999.0) &
            (rainf_data != -9999.0) &
            (crasm_data != -9999.0) &
            (tair_data != -9999.0) &
            (dem_data != -9999.0) &
            (clay_data != -9999.0) &
            ((clay_data <=100)&(clay_data >= -9)) &
            (sand_data != -9999.0) &
            (silt_data != -9999.0) &
            (xpos_data != -9999.0) &
            (ypos_data != -9999.0)
        )

        # Find valid value locations
        valid_loc = np.where(
            (~np.isnan(cci_data)) &
            (cci_data != -9999.0)&
            (cci_data > 0) &
            (cci_data <= 1)
        )

        if len(loc[0]) > 0:
            # Prepare prediction data
            features = np.column_stack((
                ndvi_data[loc],
                albedo_data[loc],
                surft_data[loc],
                rainf_data[loc],
                crasm_data[loc],
                tair_data[loc],
                dem_data[loc],
                clay_data[loc],
                sand_data[loc],
                silt_data[loc],
                date_data[loc]
            ))

            # Load model and predict
            xgboost_model = xgb.Booster(model_file=model_file)
            predict_data = xgb.DMatrix(features)
            results = xgboost_model.predict(predict_data)

            # Create output array
            output = np.full((im_height, im_width), -9999.0, dtype='float32')

            # Process prediction results
            results = np.clip(results, 0.02, 1.0)  # Limit predicted values to valid range
            output[loc] = results

            # Keep original valid values
            if len(valid_loc[0]) > 0:
                output[valid_loc] = cci_data[valid_loc]

            # Output statistics
            log_message = (
                f'Processing: {current_date} ({month_name})\n'
                f'Total pixels: {im_height * im_width}\n'
                f'Predicted pixels: {len(loc[0])}\n'
                f'Valid original pixels: {len(valid_loc[0])}\n'
                f'Missing pixels after prediction: {np.sum(output == -9999.0)}\n'
                f'-----------------------------------\n')
            print(log_message)
            log_file.write(log_message)

            # Save results
            driver = gdal.GetDriverByName('GTiff')
            data_frame = driver.Create(
                os.path.join(out_path, f'{current_date}.tif'),
                im_width,
                im_height,
                1,
                gdal.GDT_Float32
            )
            data_frame.SetProjection(im_projection)
            data_frame.SetGeoTransform(im_geotransform)
            data_frame.GetRasterBand(1).WriteArray(output)
            data_frame.GetRasterBand(1).SetNoDataValue(-9999)

            print(f'Saved: {current_date}.tif\n')

            # Clean memory
            del driver, data_frame, predict_data, results, output
        else:
            log_message = f'No pixels to predict for date: {current_date}\n'
            print(log_message)
            log_file.write(log_message)

    except Exception as e:
        error_message = f'Error processing {current_date}: {str(e)}\n'
        print(error_message)
        log_file.write(error_message)

end_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
final_message = (
    '\nPrediction completed\n'
    f'Start time: {start_time}\n'
    f'End time: {end_time}\n'
)
print(final_message)
log_file.write(final_message)
log_file.close()