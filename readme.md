# BPJS Kesehatan Hackathon Cost Prediction
##### Datasets :
![Kaggle](https://img.shields.io/badge/Kaggle-035a7d?style=for-the-badge&logo=kaggle&logoColor=white)
https://www.kaggle.com/datasets/bagusbpg/bpjs-kesehatan-hackathon-2021-cost-prediction
![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)
The app is here :
https://yodialfa-bpjscostprediction-hackathon-cost-prediction-lm7sx5.streamlit.app/ 

## Data Understanding
In this dataset we have 36 columns, and 57.971, but I don't know description per each columns because the author was encoded the features.
Some data we can know the description, like tglpelayanan, peserta, case, and unit_cost."tglpelayanan is date of serve, peserta is encoded of member, case total case of serve, unit cost is the target"

## Data Cleansing
In this section I drop some features like rowid which number of row, tglpelayanan, peserta and case. And when I check missing value I didn't found missing value."

## Baseline Modelling

I split the data to train and test with ratio 80/20, then modelling with baseline model (Linear Regression, Ridge and Lasso). 

And I have r2 score is 0.89 and rmse is 607.313 (Multiple Linear Regression), 
Ridge is 0.89 and rmse is 607.355 (Ridge),
And with Lasso r2 score is 0.89 too, and RMSE is 607.315.

I try Decission Tree, Random Forest and LightGBM without hyper parameter tunning, And I got 
R2 Score and RMSE (0.97 , 272.031 (DT),
R2 Score 0.98 RMSE = 229.257 (Random Forest), 
R2 Score = 0.976 Rmse = 283.011 (Light GBM). 

It's good than Linear Regression, Ridge and Lasso".

## Deep Cleansing
After using baseline for our data. I try to cleaning the data again. First of all I drop 'ds' because has 1 value, and I drop 'peserta'.Because member id isn't using for modelling.
And I see on target (unit_cost) didn't normal. So I try to doing remove outlier for the next step. But after we doing removing outlier, the target still didn't normal. So I doing Feature Engineering using Log Transformation. And After that the data is normal, but bimodal(two peak). So I try to going the next step because some ML has immune that bimodal data.

Log Transformation :
```
#transformation data to reduce outlier
final_df['unit_cost'] = np.log10(df['unit_cost'])
```

## Modelling And Development
After deep cleansing, I try to fit using some model and without tunning hyperparameter. And I got value like this :
## Before Tunning Hyperparameter
##### Decission Tree Regressor Before Tunning
- Decission Tree : R2 Score = 0.99 RMSE = 0.057609
##### Random Forest Regressor Before Tunning
- Random Forest : R2 Score = 0.99 RMSE = 0.057409
##### RANSAC Regressor Before Tunning
- Ransac Regressor : R2 Score = 0.976 RMSE = 0.0912
##### Light GBM Before Tunning
- Light GBM : R2 Score =0.986 RMSE = 0.068224
##### XGB Regressor Before Tunning
- XGB Regressor : R2 Score = 0.989 RMSE 0.06007

### After Tunning Hyperparmeter
And I try to improve that model with Hyperparameter Tunning. So I have score like this
##### Decission Tree Regressor After Tunning Hyperparameter
- Decission Tree : R2 Score = 0.973 RMSE = 0.0966
##### Random Forest Regressor After Tunning Hyperparameter
- Random Forest : R2 Score = 0.988 RMSE = 0.06575
##### RANSAC Regressor After Tunning Hyperparameter
- Ransac Regressor : R2 Score = 0.976 RMSE = 0.09106
##### Light GBM After Tunning Hyperparameter
- Light GBM : R2 Score =0.989 RMSE = 0.06044
##### XGB Regressor After Tunning Hyperparameter
- XGB Regressor : R2 Score = 0.99 RMSE 0.05686

We have some result from the model, before hyperparameter tunning DT and RF we got slowest rmse, but after tunning the rmse is high. So we can assumed that the model is overfitting.  Ransac Regressor after tunning hyperparameter show improvements, but the RMSE still high.
LGBM and XGB Regressor seem has improvement after Tunning Hyperparmeter. But between that model, XGB is the slowest RMSE than LGBM And I'll choose XGB for the model because the model is the best than other.

Before development I invers the target into normal value :
```
final_df['unit_cost'] = 10 ** final_df['unit_cost']

#split feature and target data
X = final_df.drop('unit_cost', axis=1)
y = final_df['unit_cost']

#define var of split result
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

#define best parameter
best_n_estimator = rc_xgb.best_params_['n_estimators']
best_min_child_weight = rc_xgb.best_params_['min_child_weight']
best_max_depth = rc_xgb.best_params_['max_depth']
best_learning_rate = rc_xgb.best_params_['learning_rate']
best_booster = rc_xgb.best_params_['booster']


xgb_best = XGBRegressor(n_estimators = best_n_estimator,
                        min_child_weight = best_min_child_weight,
                        max_depth = best_max_depth,
                        learning_rate = best_learning_rate,
                        booster = best_booster)
#training data
xgb_best.fit(X_train, y_train)
```
Export model for development :
```
#using joblib library
import joblib
joblib.dump(xgb_best,"../BPJS_CostPrediction_xgb.pkl")

#Using save_model()
import xgboost as xgb
xgb_best.save_model("../BPJS_CostPrediction_xgb.txt")
```

Then development on streamlit.




