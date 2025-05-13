# Bias Correction and Gap Filling Workflow
This project includes six Python scripts for soil moisture dataset bias correction (e.g., CCI), employing both CDF correction and XGBoost-based correction, as well as for gap filling. The main objective is to adjust CCI data to be consistent with SMAP data. The workflow is divided into two main parts: bias correction and gap filling .

1. Bias Correction Workflow

This part applies both CDF correction and XGBoost correction to adjust CCI data to match SMAP.

Pixel Correspondence Mapping

Script: 01_CDF_correspondence.py

Establishes the one-to-one correspondence between CCI and SMAP pixels from 2015 to 2022.

Applying CDF Correction

Script: 02_Apply_CDF.py

Applies the pixel correspondence from the previous step to the entire CCI dataset (1981–2022) to obtain CDF-corrected results.

Data Format Conversion

Script: 03_Tif2Hdf5.py

Converts the CDF-corrected results (from step 2) together with auxiliary datasets (e.g., ERA5-LAND) to HDF5 format for subsequent model training.

XGBoost Model Training

Script: 04_Bias_correction_XGBoost_Train.py

Trains the XGBoost model using the HDF5-formatted dataset.

XGBoost Bias Correction Prediction

Script: 05_Bias_correction_XGBoost_predict.py

Uses the trained model to further correct the CDF-corrected data, producing the final bias-corrected results.



2. Gap Filling Workflow

This part fills gaps in the bias-corrected soil moisture data using auxiliary datasets.

Data Format Conversion

Script: 03_Tif2Hdf5.py

Converts the bias-corrected results and auxiliary datasets into HDF5 format for gap-filling model training.

XGBoost Model Training for Gap Filling

Script: 04_Bias_correction_XGBoost_Train.py

Trains an XGBoost model based on HDF5 data that includes samples with missing values.

Gap Filling Prediction

Script: 06_Gap_filling_XGBoost_predict.py

Uses the trained model to predict and fill the missing values, generating a complete soil moisture dataset.
