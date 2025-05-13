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
    :param s1: Start time string, format: '1979-01'
    :param s2: End time string, format: 'YYYY-MM'
    :return: Number of months between s1 and s2.
    """
    s1_month = int(s1.split('-')[1])
    s2_month = int(s2.split('-')[1])
    s1_year = int(s1.split('-')[0])
    s2_year = int(s2.split('-')[0])
    return 13 - s1_month + s2_month + (s2_year - s1_year - 1) * 12

def unify_period(files_list: list, s: str, e: str):
    """
    :param files_list: List of files for a factor
    :param s: Start date, format: 'YYYYMMDD'
    :param e: End date, format: 'YYYYMMDD'
    :return: Filtered file list within the period.
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
    Convert date string to month abbreviation
    :param date_str: Date string (YYYYMMDD)
    :return: Month abbreviation (Jan, Feb, etc.)
    """
    month_map = {
        '01': 'Jan', '02': 'Feb', '03': 'Mar', '04': 'Apr',
        '05': 'May', '06': 'Jun', '07': 'Jul', '08': 'Aug',
        '09': 'Sep', '10': 'Oct', '11': 'Nov', '12': 'Dec'
    }
    month = date_str[4:6]
    return month_map[month]

def check_data_completeness(file_lists):
    """Check if all variables have data for each date."""
    dates = set()
    for files in file_lists:
        current_dates = {os.path.basename(f).split('.')[0] for f in files}
        dates.update(current_dates)
    complete_dates = []
    for date in sorted(dates):
        has_all_data = all(
            any(date == os.path.basename(f).split('.')[0] for f in files)
            for files in file_lists
        )
        if has_all_data:
            complete_dates.append(date)
    return complete_dates

def filter_files_by_dates(files, valid_dates):
    return [f for f in files if os.path.basename(f).split('.')[0] in valid_dates]

print("Starting prediction at:", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

Ndvi_path = r"E:\GSSM\03_gap\02_wrap\01_NDVI_gimms_MOD"
Albedo_path = r"E:\GSSM\03_gap\02_wrap\02_ALBEDO"
SurfT_path = r"E:\GSSM\03_gap\02_wrap\03_soil_temperature"
Rainf_path = r"E:\GSSM\03_gap\02_wrap\07_total_precipitation_sum"
CraSM_path = r"E:\GSSM\03_gap\02_wrap\05_volumetric_soil_water_layer"
Tair_path = r"E:\GSSM\03_gap\02_wrap\06_temperature_2m"
Raina_path = r"E:\GSSM\02_BCDF\对应关系"
SCF_path = r"E:\GSSM\03_gap\02_wrap\08_Soil-Component-Fraction"
Pos_path = r"E:\GSSM\03_gap\02_wrap\09_Position"
Dem_path = r"E:\GSSM\03_gap\02_wrap\10_GTOPO30DEM"

models_path = r"E:\Quasi-SMAP\01_MLSMAP\02_model_test"
CCI_path = os.path.join(r"E:\SMAP\01nan")
out_path = r"E:\Quasi-SMAP\01_MLSMAP\03_predict_test"

user_fun.makedir(out_path)

start = '20221212'
end = '20221230'

Ndvi_files = unify_period(user_fun.file_search(Ndvi_path, '.tif'), s=start, e=end)
Albedo_files = unify_period(user_fun.file_search(Albedo_path, '.tif'), s=start, e=end)
SurfT_files = unify_period(user_fun.file_search(SurfT_path, '.tif'), s=start, e=end)
Rainf_files = unify_period(user_fun.file_search(Rainf_path, '.tif'), s=start, e=end)
CraSM_files = unify_period(user_fun.file_search(CraSM_path, '.tif'), s=start, e=end)
Tair_files = unify_period(user_fun.file_search(Tair_path, '.tif'), s=start, e=end)
Raina_files = unify_period(user_fun.file_search(Raina_path, '.tif'), s=start, e=end)
CCI_files = unify_period(user_fun.file_search(CCI_path, '.tif'), s=start, e=end)

all_file_lists = [Ndvi_files, Albedo_files, SurfT_files, Rainf_files,
                  Raina_files, CraSM_files, Tair_files]
valid_dates = check_data_completeness(all_file_lists)

Ndvi_files = filter_files_by_dates(Ndvi_files, valid_dates)
Albedo_files = filter_files_by_dates(Albedo_files, valid_dates)
SurfT_files = filter_files_by_dates(SurfT_files, valid_dates)
Rainf_files = filter_files_by_dates(Rainf_files, valid_dates)
Rainfd_files = filter_files_by_dates(Raina_files, valid_dates)
CraSM_files = filter_files_by_dates(CraSM_files, valid_dates)
Tair_files = filter_files_by_dates(Tair_files, valid_dates)

print(f"Original file count: {len(Ndvi_files)}")
print(f"Filtered file count: {len(valid_dates)}")
print(f"Missing date count: {len(Ndvi_files) - len(valid_dates)}")

SCF_files = user_fun.file_search(SCF_path, '.tif')
Pos_files = user_fun.file_search(Pos_path, '.tif')
Dem_files = user_fun.file_search(Dem_path, '.tif')

Fraction_Clay = SCF_files[0]
Fraction_Sand = SCF_files[1]
Fraction_Silt = SCF_files[2]
Pos_x = Pos_files[0]
Pos_y = Pos_files[1]
Dem = Dem_files[0]

crs_tif = CCI_files[0]
im_width = query_tif(crs_tif)[0]
im_height = query_tif(crs_tif)[1]
im_projection = query_tif(crs_tif)[2]
im_geotransform = query_tif(crs_tif)[3]

log_file = open(os.path.join(out_path, 'prediction_log.txt'), 'w')

print(f"Starting predictions from {start} to {end}")
print(f"Total files to process: {len(Ndvi_files)}")
start_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))

for day_index in range(len(Ndvi_files)):
    try:
        current_date = os.path.basename(Ndvi_files[day_index]).split('.')[0]
        month_name = get_month_name(current_date)
        model_file = os.path.join(models_path, f'{month_name}-1.model')
        if not os.path.exists(model_file):
            print(f'Model file not found: {model_file}')
            continue

        ndvi_data = query_tif(Ndvi_files[day_index])[4]
        albedo_data = query_tif(Albedo_files[day_index])[4]
        surft_data = query_tif(SurfT_files[day_index])[4]
        rainf_data = query_tif(Rainf_files[day_index])[4]
        raina_data = query_tif(Raina_files[day_index])[4]
        crasm_data = query_tif(CraSM_files[day_index])[4]
        tair_data = query_tif(Tair_files[day_index])[4]

        dem_data = query_tif(Dem)[4]
        clay_data = query_tif(Fraction_Clay)[4]
        sand_data = query_tif(Fraction_Sand)[4]
        silt_data = query_tif(Fraction_Silt)[4]
        xpos_data = query_tif(Pos_x)[4]
        ypos_data = query_tif(Pos_y)[4]

        date_data = np.full((im_height, im_width), float(current_date[:4]))

        loc = np.where(
            (~np.isnan(ndvi_data)) &
            (ndvi_data != -9999.0) &
            ((ndvi_data < 1) & (ndvi_data > -1)) &
            (~np.isnan(albedo_data)) &
            (~np.isnan(surft_data)) &
            (~np.isnan(rainf_data)) &
            (~np.isnan(crasm_data)) &
            (~np.isnan(tair_data)) &
            (~np.isnan(dem_data)) &
            (~np.isnan(clay_data)) &
            ((clay_data > -9) & (clay_data != 0)) &
            (~np.isnan(sand_data)) &
            (~np.isnan(silt_data)) &
            (~np.isnan(xpos_data)) &
            (~np.isnan(ypos_data)) &
            (~np.isnan(raina_data) & (raina_data != -9999.0))
        )

        if len(loc[0]) > 0:
            features = np.column_stack((
                ndvi_data[loc],
                albedo_data[loc],
                surft_data[loc],
                rainf_data[loc],
                raina_data[loc],
                crasm_data[loc],
                tair_data[loc],
                dem_data[loc],
                clay_data[loc],
                sand_data[loc],
                silt_data[loc],
                xpos_data[loc],
                ypos_data[loc]
            ))

            xgboost_model = xgb.Booster(model_file=model_file)
            predict_data = xgb.DMatrix(features)
            results = xgboost_model.predict(predict_data)

            output = np.full((im_height, im_width), -9999.0, dtype='float32')

            results = np.clip(results, 0.02, 1.0)
            output[loc] = results

            log_message = (
                f'Processing: {current_date} ({month_name})\n'
                f'Total pixels: {im_height * im_width}\n'
                f'Predicted pixels: {len(loc[0])}\n'
                f'Missing pixels after prediction: {np.sum(output == -9999.0)}\n'
                f'-----------------------------------\n'
            )
            print(log_message)
            log_file.write(log_message)

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
