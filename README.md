# TempTL_S1

This codes are part of research publication titled "Interseasonal Transfer Learning for Crop Mapping Using Sentinel-1 Data" by Pandzic, M. et al. Consider citing our work if you found it useful (to be published soon).
In short, we are using source data (2017-2020 seasons) and target data (2021 season) to build a crop classificaiton model based on Sentinel-1 imagery, and then transferrring this pretrained mdoels in time. Sentinel-1 data can be obtained from Alaska Satellite Facility (https://asf.alaska.edu/) and ground truth data is owned by Biosense Institute (www.biosense.rs).

We are testing three algorithms using different transferring approaches, as well as traditional approach for crop classification. Approaches are:
1) naive transfer - transferring pretrained model as such, i.e., without adaptations
2) transfer learning (at different data increments from target domain) - using pretrained model and samll portion of target season data to retrain the model
3) from scratch (historical) - using optimized hyperparameters on historical data, but target data for model training
4) from scratch (2021) - optimizing hyperparameters and training model using data from target domain only

Code cleaning in progress...

01_hypp_opt_XXX.py - Files dedicated to hyperparameter optimization for Random Forest (RF),  Convolutional Neural Network (CNN) and Transformer (TR).


02_tl_CNN_classification_dp_SKfold - Script for running all algorithms on test dataset.


Metric_estimation_test_set_per_calss - File for metrics display


BreizhCrops-script.py and breizhcrops.py - Originally taken from https://github.com/dl4sits/BreizhCrops


