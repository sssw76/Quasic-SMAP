# -*- coding: utf-8 -*-
from Drought_Event_Class import query_tif
import numpy as np
import user_fun
import time
import os
import h5py


def calculate_month(s1: str, s2: str):
    """
    :param s1: Start time string. Format: '1979-01'
    :param s2: End time string.
    :return: The number of months.
    """
    s1_month = int(s1[4:6])
    s2_month = int(s2[4:6])
    s1_year = int(s1[0:4])
    s2_year = int(s2[0:4])
    return 13 - s1_month + s2_month + (s2_year - s1_year - 1) * 12


def unify_period(files_list: list, s: str, e: str):
    """
    :param files_list: File list of factors.
    :param s: Start date in format: '1979-01'.
    :param e: As above.
    :return: Filtered file list.
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


def group_by_month(files_list: list):
    jan, feb, mar, apr, may, jun = [], [], [], [], [], []
    jul, aug, sep, oco, nov, dec = [], [], [], [], [], []
    for file in files_list:
        month = os.path.basename(file).split('.')[0][4:6]
        if month == '01':
            jan.append(file)
        elif month == '02':
            feb.append(file)
        elif month == '03':
            mar.append(file)
        elif month == '04':
            apr.append(file)
        elif month == '05':
            may.append(file)
        elif month == '06':
            jun.append(file)
        elif month == '07':
            jul.append(file)
        elif month == '08':
            aug.append(file)
        elif month == '09':
            sep.append(file)
        elif month == '10':
            oco.append(file)
        elif month == '11':
            nov.append(file)
        elif month == '12':
            dec.append(file)
    return jan, feb, mar, apr, may, jun, jul, aug, sep, oco, nov, dec


# Add date completeness check function here
def check_data_completeness(file_lists):
    """Check if all variables have data for each date"""
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


start_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
# Set input paths
Ndvi_path = r"E:\GSSM\03_gap\02_wrap\01_NDVI_gimms_MOD"
Albedo_path = r"E:\GSSM\03_gap\02_wrap\02_ALBEDO"
SurfT_path = r"E:\GSSM\03_gap\02_wrap\03_soil_temperature"
Rainf_path = r"E:\GSSM\03_gap\02_wrap\07_total_precipitation_sum"
CraSM_path = r"E:\GSSM\03_gap\02_wrap\05_volumetric_soil_water_layer"
Tair_path = r"E:\GSSM\03_gap\02_wrap\06_temperature_2m"
Rainfd_path = r"E:\GSSM\02_BCDF\对应关系"
# NDVIl30_path = r"E:\GSSM\03_gap\02_wrap\12_NDVI_lag30"
SCF_path = r"E:\GSSM\03_gap\02_wrap\08_Soil-Component-Fraction"
Pos_path = r"E:\GSSM\03_gap\02_wrap\09_Position"
Dem_path = r"E:\GSSM\03_gap\02_wrap\10_GTOPO30DEM"
Mask_path = r"E:\GSSM\03_gap\05_six_gaps\01_gap_tif_2"


folders = ['1.origin']
for folder in folders:
    CCI_path = os.path.join(r"E:\SMAP\01nan")
    # models_path = os.path.join(r"E:\Quasi-SMAP\01_MLSMAP\01_feature")
    # user_fun.makedir(models_path)
    start = '20150331'
    end = '20221231'
    month_count = calculate_month(start, end)
    Ndvi_files = unify_period(user_fun.file_search(Ndvi_path, '.tif'), s=start, e=end)
    Albedo_files = unify_period(user_fun.file_search(Albedo_path, '.tif'), s=start, e=end)
    SurfT_files = unify_period(user_fun.file_search(SurfT_path, '.tif'), s=start, e=end)
    Rainf_files = unify_period(user_fun.file_search(Rainf_path, '.tif'), s=start, e=end)
    Rainfd_files = unify_period(user_fun.file_search(Rainfd_path, '.tif'), s=start, e=end)
    CraSM_files = unify_period(user_fun.file_search(CraSM_path, '.tif'), s=start, e=end)
    Tair_files = unify_period(user_fun.file_search(Tair_path, '.tif'), s=start, e=end)
    CCI_files = unify_period(user_fun.file_search(CCI_path, '.tif'), s=start, e=end)
    # Execute date check and file filtering
    all_file_lists = [Ndvi_files, Albedo_files, SurfT_files, Rainf_files,
                      Rainfd_files, CraSM_files, Tair_files, CCI_files]
    valid_dates = check_data_completeness(all_file_lists)

    # Filter all file lists
    Ndvi_files = filter_files_by_dates(Ndvi_files, valid_dates)
    Albedo_files = filter_files_by_dates(Albedo_files, valid_dates)
    SurfT_files = filter_files_by_dates(SurfT_files, valid_dates)
    Rainf_files = filter_files_by_dates(Rainf_files, valid_dates)
    Rainfd_files = filter_files_by_dates(Rainfd_files, valid_dates)
    CraSM_files = filter_files_by_dates(CraSM_files, valid_dates)
    Tair_files = filter_files_by_dates(Tair_files, valid_dates)
    CCI_files = filter_files_by_dates(CCI_files, valid_dates)

    print(f"Original file count: {len(Ndvi_files)}")
    print(f"File count after filtering: {len(valid_dates)}")
    print(f"Number of missing dates: {len(Ndvi_files) - len(valid_dates)}")

    SCF_files = user_fun.file_search(SCF_path, '.tif')
    Pos_files = user_fun.file_search(Pos_path, '.tif')
    Dem_files = user_fun.file_search(Dem_path, '.tif')
    # Mask_files = user_fun.file_search(Mask_path, '.tif')
    Fraction_Clay = SCF_files[0]
    Fraction_Sand = SCF_files[1]
    Fraction_Silt = SCF_files[2]
    Pos_x = Pos_files[0]
    Pos_y = Pos_files[1]
    Dem = Dem_files[0]
    # Mask = Mask_files[0]

    # Information of images.
    im_width = query_tif(Dem)[0]
    im_height = query_tif(Dem)[1]
    im_projection = query_tif(Dem)[2]
    im_geotransform = query_tif(Dem)[3]

    # Grouping by month.
    Ndvi_group = group_by_month(Ndvi_files)
    Albedo_group = group_by_month(Albedo_files)
    SurfT_group = group_by_month(SurfT_files)
    Rainf_group = group_by_month(Rainf_files)
    Rainfd_group = group_by_month(Rainfd_files)
    CraSM_group = group_by_month(CraSM_files)
    Tair_group = group_by_month(Tair_files)
    CCI_group = group_by_month(CCI_files)
    print(Rainfd_group)

    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    # months = ['Jun']
    # months = ['Feb', 'Mar', 'Apr', 'May', 'Jun',
    #           'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    # months = ['May', 'Jun',
    #           'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    for k, m in enumerate(months):
        # txt_file = open(os.path.join(models_path, m + '.txt'), 'w')
        single_month_number = len(Ndvi_group[k])
        container_ndvi, container_albedo = np.array([], dtype='float32'), np.array([], dtype='float32')
        container_surft, container_rainf = np.array([], dtype='float32'), np.array([], dtype='float32')
        container_rainfd, container_crasm = np.array([], dtype='float32'), np.array([], dtype='float32')
        container_tair, container_cci = np.array([], dtype='float32'), np.array([], dtype='float32')
        container_dem, container_clay = np.array([], dtype='float32'), np.array([], dtype='float32')
        container_sand, container_silt = np.array([], dtype='float32'), np.array([], dtype='float32')
        container_xpos, container_ypos = np.array([], dtype='float32'), np.array([], dtype='float32')
        container_date, container_mask = np.array([], dtype='float32'), np.array([], dtype='float32')

        for smn in range(single_month_number):
            ndvi_data = query_tif(Ndvi_group[k][smn])[4].astype('float32')
            albedo_data = query_tif(Albedo_group[k][smn])[4].astype('float32')
            surft_data = query_tif(SurfT_group[k][smn])[4].astype('float32')
            rainf_data = query_tif(Rainf_group[k][smn])[4].astype('float32')
            rainfd_data = query_tif(Rainfd_group[k][smn])[4]
            crasm_data = query_tif(CraSM_group[k][smn])[4].astype('float32')
            tair_data = query_tif(Tair_group[k][smn])[4].astype('float32')
            cci_data = query_tif(CCI_group[k][smn])[4].astype('float32')
            dem_data = query_tif(Dem)[4].astype('float32')
            clay_data = query_tif(Fraction_Clay)[4].astype('float32')
            sand_data = query_tif(Fraction_Sand)[4].astype('float32')
            silt_data = query_tif(Fraction_Silt)[4].astype('float32')
            xpos_data = query_tif(Pos_x)[4].astype('float32')
            ypos_data = query_tif(Pos_y)[4].astype('float32')
            # date_data = np.repeat(float(os.path.basename(Ndvi_group[k][smn]).split('.')[0][0:4]),
            #                       im_width * im_height)
            # date_data.resize(im_height, im_width)
            # mask_data = query_tif(Mask)[4]

            container_ndvi = np.append(container_ndvi, ndvi_data)
            container_albedo = np.append(container_albedo, albedo_data)
            container_surft = np.append(container_surft, surft_data)
            container_rainf = np.append(container_rainf, rainf_data)
            container_rainfd = np.append(container_rainfd, rainfd_data)
            container_crasm = np.append(container_crasm, crasm_data)
            container_tair = np.append(container_tair, tair_data)
            container_cci = np.append(container_cci, cci_data)
            container_dem = np.append(container_dem, dem_data)
            container_clay = np.append(container_clay, clay_data)
            container_sand = np.append(container_sand, sand_data)
            container_silt = np.append(container_silt, silt_data)
            container_xpos = np.append(container_xpos, xpos_data)
            container_ypos = np.append(container_ypos, ypos_data)
            # container_date = np.append(container_date, date_data)
            # container_mask = np.append(container_mask, mask_data)

        for mask in range(1, 2):
            loc = np.where(
                (~np.isnan(container_ndvi)) &  # Not NaN in ndvi_data
                (container_ndvi > -1) &  # Value range constraint for ndvi
                (container_ndvi < 1) &
                (~np.isnan(container_albedo)) &  # Not NaN in albedo_data
                (~np.isnan(container_surft)) &
                (~np.isnan(container_rainf)) &
                (~np.isnan(container_crasm)) &
                (~np.isnan(container_tair)) &
                (container_rainfd > 0) &  # Value range constraint for rainfd
                (container_rainfd < 1) &
                (~np.isnan(container_cci)) &  # Not NaN in cci_data
                (container_cci > 0) &  # Value range constraint for cci
                (container_cci < 1) &
                (~np.isnan(container_dem)) &
                (~np.isnan(container_clay)) &
                (~np.isnan(container_sand)) &
                (~np.isnan(container_silt)) &
                (~np.isnan(container_xpos)) &
                (~np.isnan(container_ypos))
                # (~np.isnan(container_date))
            )

            ndvi = container_ndvi[loc].reshape(-1, 1)
            albedo = container_albedo[loc].reshape(-1, 1)
            surft = container_surft[loc].reshape(-1, 1)
            rainf = container_rainf[loc].reshape(-1, 1)
            rainfd = container_rainfd[loc].reshape(-1, 1)
            crasm = container_crasm[loc].reshape(-1, 1)
            tair = container_tair[loc].reshape(-1, 1)
            cci = container_cci[loc].reshape(-1, 1)
            dem = container_dem[loc].reshape(-1, 1)
            clay = container_clay[loc].reshape(-1, 1)
            sand = container_sand[loc].reshape(-1, 1)
            silt = container_silt[loc].reshape(-1, 1)
            xpos = container_xpos[loc].reshape(-1, 1)
            ypos = container_ypos[loc].reshape(-1, 1)
            # date = container_date[loc].reshape(-1, 1)

            factors = np.hstack((ndvi, albedo, surft, rainf, rainfd, crasm,
                                 tair, dem, clay, sand, silt, xpos, ypos))
            data_size = len(factors)
            print(data_size)
            # Save factors and cci to HDF5 file
            with h5py.File(r'E:\Quasi-SMAP\01_residual_ML_SMAP\02_features'+'/'+m+'.h5', 'w') as hf:
                hf.create_dataset('factors', data=factors)
                hf.create_dataset('cci', data=cci)
            print('Generated: '+m+'.h5')
            del factors, cci
            # # Read factors and cci
            # with h5py.File('data.h5', 'r') as hf:
            #     factors = hf['factors'][:]
            #     cci = hf['cci'][:]