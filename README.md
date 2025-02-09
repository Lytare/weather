
# Data Reconstruction and Forecasting for Meteorological Data


This repository uses the open-source "Wind Spatio-Temporal Dataset1" (https://zenodo.org/records/5516543) which contains wind speed data collected from 120 turbines in an inland wind farm over two years. 
After loading the data, the data reconstruction script removes randomized durations from a randomized selection of turbines to simulate missing data / faulty sensors. 
Missing data are then reconstructed using low-rank matrix completion (with methodology described by Rik Voorhaar) and pushed into the dataset directory of a modified branch of TimeGNN (Xu et al., 2023, https://doi.org/10.48550/arXiv.2307.14680). 
The TimeGNN_train.py script is configured to run GNN-based forecasting on either the original or reconstructed dataset. 
Both the data reconstruction and GNN methodology produce results within excellent margins.