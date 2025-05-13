# Soil Moisture Dataset Bias Correction & Gap Filling Workflow

A two-stage workflow for harmonizing CCI soil moisture data with SMAP standards and addressing data gaps using XGBoost models.

## 1. Bias Correction Workflow
*Aligns CCI (1981-2022) with SMAP (2015-2022) through dual-stage correction*

### 1.1 Pixel Correspondence Mapping
- **Script**: `01_CDF_correspondence.py`  
  Establishes spatial alignment between CCI and SMAP pixels (2015-2022 baseline)

### 1.2 Apply CDF Correction
- **Script**: `02_Apply_CDF.py`  
  Applies non-parametric CDF matching to entire CCI historical data (1981-2022)

### 1.3 Data Format Conversion
- **Script**: `03_Tif2Hdf5.py`  
  Converts outputs to HDF5 format with:
  - CDF-corrected CCI data
  - NDVI data
  - ERA5-LAND meteorological data
  - DEM data
  - HWSD data

### 1.4 XGBoost Model Training
- **Script**: `04_Bias_correction_XGBoost_Train.py`  
  Trains residual correction model using:
  - SMAP data as ground truth
  - CDF-corrected CCI as primary input
  - Auxiliary environmental features

### 1.5 XGBoost Bias Correction
- **Script**: `05_Bias_correction_XGBoost_predict.py`  
  Generates final corrected dataset (1981-2022) through model inference

## 2. Gap Filling Workflow
*Addresses missing values in bias-corrected data using environmental covariates*

### 2.1 Data Preparation
- **Script**: `03_Tif2Hdf5.py` (Reused)  
  Packages for gap filling:
  - Bias-corrected CCI with missing values
  - Full-coverage ERA5-LAND data
  - Static landscape features

### 2.2 Gap-Filling Model Training
- **Script**: `04_Bias_correction_XGBoost_Train.py` (Modified)  
  Trains on complete samples with:
  - Artificial masking for validation
  - Enhanced feature engineering
  - Temporal covariates integration

### 2.3 Gap Imputation
- **Script**: `06_Gap_filling_XGBoost_predict.py`  
  Produces gap-free soil moisture dataset through:
  - Spatial-temporal pattern learning
  - Multi-feature joint prediction
  - Uncertainty quantification
 

Key Implementation Notes:

Temporal Alignment: All scripts handle time-series synchronization between datasets

Spatial Consistency: Maintains 0.25° grid resolution throughout processing

Model Persistence: Trained XGBoost models are saved for reproducibility

Validation: Includes embedded cross-validation routines in training scripts

HDF5 Structure: Uses hierarchical storage with metadata preservation
