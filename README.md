# Bias Correction and Gap Filling Workflow
This project includes six Python scripts for bias correction of soil moisture datasets (e.g., CCI), encompassing both CDF correction and XGBoost correction, as well as gap filling. The main objective is to adjust the CCI dataset to be consistent with SMAP data. The overall workflow is divided into two parts: bias correction and gap filling.

1. Bias Correction Workflow
This part performs CDF correction and XGBoost correction to adjust CCI data to match SMAP:
Pixel Correspondence Mapping (01_CDF_correspondence.py)
Function: Establish the one-to-one correspondence between CCI and SMAP pixels from 2015 to 2022.
Applying CDF Correction (02_Apply_CDF.py)
Function: Apply the pixel correspondence from step 1 to all CCI data (1981-2022) to obtain the CDF-corrected results.
Data Format Conversion (03_Tif2Hdf5.py)
Function: Convert the CDF-corrected results (from step 2) together with auxiliary datasets (e.g., ERA5-LAND) to HDF5 format for subsequent model training.
XGBoost Model Training (04_XGBoost_Train.py)
Function: Train the XGBoost model using the HDF5 dataset.
XGBoost Bias Correction Prediction (05_Bias_correction_XGBoost_predict.py)
Function: Use the trained model to further bias-correct the CDF-corrected data and output the final corrected results.


2. Gap Filling Workflow
This part fills gaps in the bias-corrected soil moisture data using auxiliary datasets:
Data Format Conversion (03_Tif2Hdf5.py)
Function: Convert the bias-corrected results and auxiliary datasets into HDF5 format for training the gap-filling model.
XGBoost Model Training for Gap Filling (04_XGBoost_Train.py)
Function: Train XGBoost model based on HDF5 data including missing samples.
Gap Filling Prediction (06_Gap_filling_XGBoost_predict.py)
Function: Use the trained model to predict and fill the missing values, generating a complete soil moisture dataset.
