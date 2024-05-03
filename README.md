<center>
<img src=https://st2.depositphotos.com/3955303/5656/i/450/depositphotos_56566631-stock-photo-abalone-shell.jpg alt="Abalone" style="width:400px;">

# <center> **Regression with an Abalone Dataset**
### <center> Playground Series - Season 4, Episode 4 (<a href=(https://www.kaggle.com/competitions/playground-series-s4e4/overview >Source: Kaggle</a>)


**Project Overview:** The goal of this kaggle challenge is to predict the age of abalones from various physical measurements. The dataset provided (both train and test) were generated from a deep learning model trained on <a href=(https://archive.ics.uci.edu/dataset/1/abalone >the original Abalone dataset</a>. Feature distributions are close to, but not exactly the same, as the original.  

_____________________________________________________________________________________________________________________________________________
</center>

**Files and folders in this repository:** 
 - EDA.ipynb: exploratory data analysis of datasets provided
 - EDA-original.ipynb: exploratory data analysis of the original abalone dataset from UCL
 - preprocessing.ipynb: preprocessing, feature engineering, and feature selection performed to make the model selection more efficient
 - model_selection.ipynb: testing of several ML models to see what works best in this challenge
 - final_predictions.ipynb: predictions made with a CatBoostRegressor, which landed 21% of the public leaderboard
 - kaggle-API.ipynb:     download the latest competition data and make submissions using the Kaggle API
 - data: folder where we store all datasets (raw and processed)



 _______________________________

 
# **Results** 
In general, regression models (e.g., RandomForestRegressor and XGBRegressor) were better at minimizing the RMSLE and hence at winning this competition. However, classification models resulted in "better" predictions as they were able to predict the age of very young and old abalones (low and high number of rings, respectively), while regressors made predictions mostly in the mid range (5 to 20 Rings).

**The best predictions here were made with a CatBoostRegressor, which achieved a public Kaggle score (RMSLE) of 0.14671, landing (at the time of submission) number 422 out of 2,006 (21%) of the public leaderboard.**

# **Recomendations**

In order to improve the score, the following steps should be considered
1. Feature Engineering: enginnering more features (e.g., volume, clusters)
2. Hyperparameter Tuning: The model should be tuned to obtain the best hyperparameters
3. Ensemble model: ensembling or stacking multiple models could be attempted