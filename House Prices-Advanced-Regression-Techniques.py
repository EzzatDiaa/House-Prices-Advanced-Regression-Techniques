# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 11:58:58 2022

@author: EzzatAbdelnaby
"""

# Data wrangling
import pandas as pd
import numpy as np

# Data visualisation
import matplotlib.pyplot as plt
import seaborn as sns

# Statistics
from scipy.stats import norm, boxcox_normmax
from scipy.special import boxcox1p

# Modelling
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LassoCV, RidgeCV, ElasticNetCV
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.svm import SVR

# Remove warnings
import warnings
warnings.filterwarnings('ignore')

# Miscellaneous
from collections import Counter


### IMPORT AND READ DATA
train = pd.read_csv("C:/Users/zezod/OneDrive/Desktop/House Prices-Advanced-Regression-Techniques/train.csv")
test = pd.read_csv("C:/Users/zezod/OneDrive/Desktop/House Prices-Advanced-Regression-Techniques/test.csv")
ss = pd.read_csv("C:/Users/zezod/OneDrive/Desktop/House Prices-Advanced-Regression-Techniques/sample_submission.csv") 
## LOOKING INTO DATA
print(train.head())
print(test.head())
print(ss.head())


### Exploratory data analysis(EDA)
print(train.describe().transpose())
print(test.describe().transpose())

## Combine training and test set
combine = pd.concat([train, test])

##Check data types
print(combine.dtypes.value_counts())

## Summary statistics for sale price
print(train['SalePrice'].describe())

## Sale price distribution
plt.figure(figsize = (12, 5))
sns.set_style('white')
sns.distplot(train['SalePrice'], label = 'Skewness: %.2f'%train['SalePrice'].skew())
plt.legend(loc = 'best')
plt.title('Sale Price Distribution')


## Skewness and kurtosis
print("Skewness: %f"%train['SalePrice'].skew())
print("Kurtosis: %f"%train['SalePrice'].kurt())

### Correlation between numerical variables
corr = combine.corr()
plt.figure(figsize = (12, 8))
sns.heatmap(corr, cmap = 'coolwarm')
#Just by eyeballing the heatmap, we can observe signs of multicollinearity between some features.

## Features most correlated with sale price
print(corr['SalePrice'].sort_values(ascending = False).head(10))


## Sale price against overall quality 
sns.set_style('white')
sns.factorplot(x = 'OverallQual', y = 'SalePrice', data = train, kind = 'bar')
plt.title('Sale Price Against Overall Quality')
#Sale price increases with overall quality

## Sale price against GrLivArea
sns.set_style('darkgrid')
sns.scatterplot(x = 'GrLivArea', y = 'SalePrice', data = train)
plt.title('Sale Price Against GrLivArea')
# there are two outliers in the scatter plot with large GrLivArea but low sale price. i need to address this later on by removing them from the training set.

##Sale price against garage cars
sns.set_style('white')
sns.factorplot(x = 'GarageCars', y = 'SalePrice', data = train, kind = 'bar')
plt.title('Sale Price Against Garage Cars')
#Houses with 3 garage cars have the highest sale price.

## Sale price against TotalBsmtSF
sns.set_style('darkgrid')
sns.scatterplot(x = 'TotalBsmtSF', y = 'SalePrice', data = train)
plt.title('Sale Price Against TotalBsmtSF')
#Another outlier! The data point on the far right has a large TotalBsmtSF but a low sale price.

## Sale price against FullBath
sns.set_style('white')
sns.factorplot(x = 'FullBath', y = 'SalePrice', data = train, kind = 'bar')
plt.title('Sale Price Against Full Bath')

## Sale price against year built
sns.set_style('white')
sns.lineplot(x = 'YearBuilt', y = 'SalePrice', data = train)
plt.title('Sale Price Against Year Built')


## Mega scatter plot
sns.set_style('darkgrid')
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(train[cols])



### Examine missing data
## Missing data in combined dataframe
missing = combine.isnull().sum()
missing = missing[missing > 0]

##Percentage missing
percent_missing = missing / len(combine)

## Concat missing and percentage missing
missing_df = pd.concat([missing, percent_missing], axis = 1, keys = ['Total', 'Percent'])

## Drop sale price because test set does not have sale price
missing_df = missing_df.drop('SalePrice')

## Create dataframe for missing data 
missing_df.sort_values(by = 'Total', ascending = False, inplace = True)
print(missing_df)

## Columns with missing data
plt.figure(figsize = (15, 5))
missing_df['Total'].plot(kind = 'bar')
print("Number of features with missing data in combined dataframe: ", len(missing_df))
missing_features = missing_df.index
print(missing_features)

###Missing values in garage
## Get missing features that are related to garage
garage_features = [feature for feature in missing_features if 'Garage' in feature]
print(garage_features)

## Check data types of garage features
print(combine[garage_features].dtypes)
# 3 numerical garage features and 4 categorical garage features.
#I will fill the numerical garage features with 0 and the categorcial garage features with None.

## Fill missing garage features
for feature in garage_features:
    if combine[feature].dtype == 'object':
        combine[feature] = combine[feature].fillna('None')
    else:
        combine[feature] = combine[feature].fillna(0)
        
 ## Get missing features that are related to basement
basement_features = [feature for feature in missing_features if 'Bsmt' in feature]
print(basement_features)

## Check data types of basement features
print(combine[basement_features].dtypes)

## Fill missing basement features
for feature in basement_features:
    if combine[feature].dtype == 'object':
        combine[feature] = combine[feature].fillna('None')
    else:
        combine[feature] = combine[feature].fillna(0)
        
## Get missing features that are related to masonry veneer
mv_features = [feature for feature in missing_features if 'MasVnr' in feature]
print(mv_features)
##fill misiing 
for feature in mv_features:
    if combine[feature].dtype == 'object':
        combine[feature] = combine[feature].fillna('None')
    else:
        combine[feature] = combine[feature].fillna(0)      
        
## there are some features have so  many missing values so i choosed to fill them by None
other_features = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu']
for feature in other_features:
    combine[feature] = combine[feature].fillna('None')  

## checking missing values again
missing = combine.isnull().sum()
missing = missing[missing > 0]
missing = missing.drop('SalePrice')
missing.sort_values(ascending = False, inplace = True)
print(missing)
for feature in missing_features:
    combine[feature] = combine[feature].fillna(combine[feature].mode()[0])
## Make sure there is no more missing data
print(combine.drop('SalePrice', axis = 1).isnull().sum().max())    
print(combine.head())


### Detect and remove outliers
train = combine[:len(train)]
test = combine[len(train):]
test = test.drop('SalePrice', axis = 1)
print("Training set shape: ", train.shape)
print("Test set shape: ", test.shape)

## Get numerical features from training set 
numerical_features = [feature for feature in train if train[feature].dtype != 'object']

## Remove Id and SalePrice
numerical_features.remove('Id')
numerical_features.remove('SalePrice')
print(numerical_features)

## Detect outliers using Tukey method
def detect_outliers(df, n, features):
    outlier_indices = [] 
    for col in features: 
        Q1 = np.percentile(df[col], 25)
        Q3 = np.percentile(df[col], 75)
        IQR = Q3 - Q1
        outlier_step = 1.5 * IQR 
        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step)].index
        outlier_indices.extend(outlier_list_col) 
    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list(key for key, value in outlier_indices.items() if value > n) 
    return multiple_outliers

outliers_to_drop = detect_outliers(train, 6, numerical_features)
print(f"We are dropping {len(outliers_to_drop)} observations.")
print(train.iloc[outliers_to_drop])

## Drop outliers and reset index
print(f"Before: {len(train)} rows")
train = train.drop(outliers_to_drop).reset_index(drop = True)
print(f"After: {len(train)} rows")

### log transformation to sale price

##Create axes to draw plots
fig, ax = plt.subplots(1, 2)

## Plot original SalePrice distribution
sns.distplot(train['SalePrice'], fit = norm, label = 'Skewness: %.2f'%train['SalePrice'].skew(), ax = ax[0])
ax[0].legend(loc = 'best')
ax[0].title.set_text('Sale Price Before Transformation')

## Apply box-cox transformation
train['SalePrice'] = np.log1p(train['SalePrice'])

##Plot transformed SalePrice distribution
sns.distplot(train['SalePrice'], fit = norm, label = 'Skewness: %.2f'%train['SalePrice'].skew(), ax = ax[1])
ax[1].legend(loc = 'best')
ax[1].title.set_text('Sale Price After Transformation')

##Rescaling the subplots
fig.set_figheight(5)
fig.set_figwidth(15)

### Box-Cox transformation to numerical features with high skewness

## Combine training and test set 
combine = pd.concat([train, test])
print("Training set shape: ", train.shape)
print("Test set shape: ", test.shape)
print("Combined shape: ", combine.shape)

## skew features
skew_features = combine[numerical_features].apply(lambda x: x.skew()).sort_values(ascending = False)
high_skew = skew_features[skew_features > 0.5]
print(f"There are {len(high_skew)} numerical features with skew greater than 0.5. ")
print(high_skew)

## Normalise skewed features
for feature in high_skew.index:
    combine[feature] = boxcox1p(combine[feature], boxcox_normmax(combine[feature] + 1))
    
## I am creating 17 new features
print(f"Before: {combine.shape[1]} columns")

combine['UnfBsmt'] = (combine['BsmtFinType1'] == 'Unf') * 1
combine['HasWoodDeck'] = (combine['WoodDeckSF'] == 0) * 1
combine['HasOpenPorch'] = (combine['OpenPorchSF'] == 0) * 1
combine['HasEnclosedPorch'] = (combine['EnclosedPorch'] == 0) * 1
combine['Has3SsnPorch'] = (combine['3SsnPorch'] == 0) * 1
combine['HasScreenPorch'] = (combine['ScreenPorch'] == 0) * 1
combine['YearsSinceRemodel'] = combine['YrSold'].astype(int) - combine['YearRemodAdd'].astype(int)
combine['TotalHomeQuality'] = combine['OverallQual'] + combine['OverallCond']
combine['TotalSF'] = combine['TotalBsmtSF'] + combine['1stFlrSF'] + combine['2ndFlrSF']
combine['YearBuiltAndRemodel'] = combine['YearBuilt'] + combine['YearRemodAdd']
combine['TotalBathrooms'] = combine['FullBath'] + combine['BsmtFullBath'] + 0.5 * (combine['HalfBath'] + combine['BsmtHalfBath'])
combine['TotalPorchSF'] = combine['OpenPorchSF'] + combine['3SsnPorch'] + combine['EnclosedPorch'] + combine['ScreenPorch'] + combine['WoodDeckSF']
combine['HasPool'] = combine['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
combine['Has2ndFloor'] = combine['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
combine['HasGarage'] = combine['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
combine['HasBsmt'] = combine['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
combine['HasFireplace'] = combine['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)

print(f"After: {combine.shape[1]} columns")
      
## Encode categorical features
print("Before: ", combine.shape)
combine = pd.get_dummies(combine)
print("After: ", combine.shape)
 
### Get the new training and test set    
train = combine[:len(train)]
test = combine[len(train):]
X_train = train.drop(['Id', 'SalePrice'], axis = 1)
Y_train = train['SalePrice']
X_test = test.drop(['Id', 'SalePrice'], axis = 1)
test_id = test['Id']
print("X_train shape: ", X_train.shape)
print("Y_train shape: ", Y_train.shape)
print("X_test shape: ", X_test.shape)    

### Define cross-validation strategy and evaluation metric

## Cross validation
kfolds = KFold(n_splits = 10, shuffle = True, random_state = 42)
# Evaluation metric

def cv_rmse(model, X = X_train, y = Y_train):
    rmse = np.sqrt(-cross_val_score(model, X, y, scoring = 'neg_mean_squared_error', cv = kfolds))
    return rmse
    
##Instantiate regressors

ridge = make_pipeline(RobustScaler(), RidgeCV())
lasso = make_pipeline(RobustScaler(), LassoCV())
elasticnet = make_pipeline(RobustScaler(), ElasticNetCV())
rf = RandomForestRegressor()
svr = SVR()
gbr = GradientBoostingRegressor()

models = [ridge, lasso, elasticnet, rf, svr, gbr]
mean = []
std = []
for model in models:
    mean.append(cv_rmse(model).mean())
    std.append(cv_rmse(model).std())

models_df = pd.DataFrame({'Model': ['Ridge', 'Lasso', 'Elastic Net', 'Random Forest', 'SVR', 'Gradient Boosting'],
                         'Mean': mean, 'Std': std})
models_df.sort_values(by = 'Mean', inplace = True, ignore_index = True)
print(models_df)

### Make predictions on test data using Ridge
ridge_model = ridge.fit(X_train, Y_train)
Y_pred = ridge_model.predict(X_test)
print(len(Y_pred))
print(ss.head())
ridge_submission = pd.DataFrame({'Id': test_id, 'SalePrice': np.expm1(Y_pred)})
print(ridge_submission.head())

## Save ridge submission
ridge_submission.to_csv("D:\House Prices-Advanced-Regression-Techniques/ridge_submission.csv", index = False)
