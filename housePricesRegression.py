import pandas as pd
import numpy as np

dirName = '/home/jahnavi/Downloads/house-prices-advanced-regression-techniques/'
trainRaw = pd.read_csv(dirName+'train.csv')
testRaw = pd.read_csv(dirName+'test.csv')

print(trainRaw.columns)


def convertNull(row, col):
    if row[col] == 'NA':
        return -1
    return row[col]

def convertUtilities(row):
    if row['Utilities'] == 'AllPub':
        return 0
    return 1

trainCapData = trainRaw
testCapData = testRaw

trainCapData.Neighborhood = trainRaw.Neighborhood.astype("category").cat.codes
print(trainCapData.Neighborhood)
trainCapData.BldgType = trainRaw.BldgType.astype("category").cat.codes
trainCapData.RoofStyle = trainRaw.RoofStyle.astype("category").cat.codes
trainCapData.Exterior2nd = trainRaw.Exterior2nd.astype("category").cat.codes
trainCapData.ExterQual = trainRaw.ExterQual.astype("category").cat.codes
trainCapData.BsmtQual = trainRaw.BsmtQual.astype("category").cat.codes
trainCapData.BsmtFinType1 = trainRaw.BsmtFinType1.astype("category").cat.codes
trainCapData.Heating = trainRaw.Heating.astype("category").cat.codes
trainCapData.Electrical = trainRaw.Electrical.astype("category").cat.codes
trainCapData.FireplaceQu = trainRaw.FireplaceQu.astype("category").cat.codes
trainCapData.GarageFinish = trainRaw.GarageFinish.astype("category").cat.codes
trainCapData.GarageQual = trainRaw.GarageQual.astype("category").cat.codes
trainCapData.PoolQC = trainRaw.PoolQC.astype("category").cat.codes
trainCapData.SaleType = trainRaw.SaleType.astype("category").cat.codes
trainCapData.LotShape = trainRaw.LotShape.astype("category").cat.codes
trainCapData.LotConfig = trainRaw.LotConfig.astype("category").cat.codes
trainCapData.Condition1 = trainRaw.Condition1.astype("category").cat.codes
trainCapData.HouseStyle = trainRaw.HouseStyle.astype("category").cat.codes
trainCapData.RoofMatl = trainRaw.RoofMatl.astype("category").cat.codes
trainCapData.MasVnrType = trainRaw.MasVnrType.astype("category").cat.codes
trainCapData.ExterCond = trainRaw.ExterCond.astype("category").cat.codes
trainCapData.BsmtCond = trainRaw.BsmtCond.astype("category").cat.codes
trainCapData.HeatingQC = trainRaw.HeatingQC.astype("category").cat.codes
trainCapData.Functional = trainRaw.Functional.astype("category").cat.codes
trainCapData.GarageType = trainRaw.GarageType.astype("category").cat.codes
trainCapData.GarageCond = trainRaw.GarageCond.astype("category").cat.codes
trainCapData.Fence = trainRaw.Fence.astype("category").cat.codes
trainCapData.MoSold = trainRaw.MoSold.astype("category").cat.codes
trainCapData.SaleCondition = trainRaw.SaleCondition.astype("category").cat.codes
trainCapData.MSZoning = trainRaw.MSZoning.astype("category").cat.codes
trainCapData.Street = trainRaw.Street.astype("category").cat.codes
trainCapData.LandContour = trainRaw.LandContour.astype("category").cat.codes
trainCapData.LandSlope = trainRaw.LandSlope.astype("category").cat.codes
trainCapData.Condition2 = trainRaw.Condition2.astype("category").cat.codes
trainCapData.Exterior1st = trainRaw.Exterior1st.astype("category").cat.codes
trainCapData.Foundation = trainRaw.Foundation.astype("category").cat.codes
trainCapData.BsmtExposure = trainRaw.BsmtExposure.astype("category").cat.codes
trainCapData.BsmtFinType2 = trainRaw.BsmtFinType2.astype("category").cat.codes
trainCapData.CentralAir = trainRaw.CentralAir.astype("category").cat.codes
trainCapData.BsmtFullBath = trainRaw.BsmtFullBath.astype("category").cat.codes
trainCapData.KitchenQual = trainRaw.KitchenQual.astype("category").cat.codes
trainCapData.PavedDrive = trainRaw.PavedDrive.astype("category").cat.codes
trainCapData.MiscFeature = trainRaw.MiscFeature.astype("category").cat.codes
trainCapData.Alley = trainRaw.Alley.astype("category").cat.codes
trainCapData.Utilities = trainRaw.Utilities.astype("category").cat.codes

trainCapData.LotFrontage = np.where(pd.isnull(trainRaw.LotFrontage), -1, trainRaw['LotFrontage'])
trainCapData.MasVnrArea = np.where(pd.isnull(trainRaw.MasVnrArea), -1, trainRaw['MasVnrArea'])
trainCapData.GarageYrBlt = np.where(pd.isnull(trainRaw.GarageYrBlt), -1, trainRaw['GarageYrBlt'])
trainCapData.BsmtFinSF1 = np.where(pd.isnull(trainRaw.BsmtFinSF1), -1, trainRaw['BsmtFinSF1'])
trainCapData.BsmtFinSF2 = np.where(pd.isnull(trainRaw.BsmtFinSF2), -1, trainRaw['BsmtFinSF2'])
trainCapData.BsmtUnfSF = np.where(pd.isnull(trainRaw.BsmtUnfSF), -1, trainRaw['BsmtUnfSF'])
trainCapData.TotalBsmtSF = np.where(pd.isnull(trainRaw.TotalBsmtSF), -1, trainRaw['TotalBsmtSF'])
trainCapData.BsmtHalfBath = np.where(pd.isnull(trainRaw.BsmtHalfBath), -1, trainRaw['BsmtHalfBath'])
trainCapData.GarageCars = np.where(pd.isnull(trainRaw.GarageCars), -1, trainRaw['GarageCars'])
trainCapData.GarageArea = np.where(pd.isnull(trainRaw.GarageArea), -1, trainRaw['GarageArea'])

print(trainCapData.LotFrontage[7])

testCapData.Neighborhood = testRaw.Neighborhood.astype("category").cat.codes
# print(trainCapData.Neighborhood)
testCapData.BldgType = testRaw.BldgType.astype("category").cat.codes
testCapData.RoofStyle = testRaw.RoofStyle.astype("category").cat.codes
testCapData.Exterior2nd = testRaw.Exterior2nd.astype("category").cat.codes
testCapData.ExterQual = testRaw.ExterQual.astype("category").cat.codes
testCapData.BsmtQual = testRaw.BsmtQual.astype("category").cat.codes
testCapData.BsmtFinType1 = testRaw.BsmtFinType1.astype("category").cat.codes
testCapData.Heating = testRaw.Heating.astype("category").cat.codes
testCapData.Electrical = testRaw.Electrical.astype("category").cat.codes
testCapData.FireplaceQu = testRaw.FireplaceQu.astype("category").cat.codes
testCapData.GarageFinish = testRaw.GarageFinish.astype("category").cat.codes
testCapData.GarageQual = testRaw.GarageQual.astype("category").cat.codes
testCapData.PoolQC = testRaw.PoolQC.astype("category").cat.codes
testCapData.SaleType = testRaw.SaleType.astype("category").cat.codes
testCapData.LotShape = testRaw.LotShape.astype("category").cat.codes
testCapData.LotConfig = testRaw.LotConfig.astype("category").cat.codes
testCapData.Condition1 = testRaw.Condition1.astype("category").cat.codes
testCapData.HouseStyle = testRaw.HouseStyle.astype("category").cat.codes
testCapData.RoofMatl = testRaw.RoofMatl.astype("category").cat.codes
testCapData.MasVnrType = testRaw.MasVnrType.astype("category").cat.codes
testCapData.ExterCond = testRaw.ExterCond.astype("category").cat.codes
testCapData.BsmtCond = testRaw.BsmtCond.astype("category").cat.codes
testCapData.HeatingQC = testRaw.HeatingQC.astype("category").cat.codes
testCapData.Functional = testRaw.Functional.astype("category").cat.codes
testCapData.GarageType = testRaw.GarageType.astype("category").cat.codes
testCapData.GarageCond = testRaw.GarageCond.astype("category").cat.codes
testCapData.Fence = testRaw.Fence.astype("category").cat.codes
testCapData.MoSold = testRaw.MoSold.astype("category").cat.codes
testCapData.SaleCondition = testRaw.SaleCondition.astype("category").cat.codes
testCapData.MSZoning = testRaw.MSZoning.astype("category").cat.codes
testCapData.Street = testRaw.Street.astype("category").cat.codes
testCapData.LandContour = testRaw.LandContour.astype("category").cat.codes
testCapData.LandSlope = testRaw.LandSlope.astype("category").cat.codes
testCapData.Condition2 = testRaw.Condition2.astype("category").cat.codes
testCapData.Exterior1st = testRaw.Exterior1st.astype("category").cat.codes
testCapData.Foundation = testRaw.Foundation.astype("category").cat.codes
testCapData.BsmtExposure = testRaw.BsmtExposure.astype("category").cat.codes
testCapData.BsmtFinType2 = testRaw.BsmtFinType2.astype("category").cat.codes
testCapData.CentralAir = testRaw.CentralAir.astype("category").cat.codes
testCapData.BsmtFullBath = testRaw.BsmtFullBath.astype("category").cat.codes
testCapData.KitchenQual = testRaw.KitchenQual.astype("category").cat.codes
testCapData.PavedDrive = testRaw.PavedDrive.astype("category").cat.codes
testCapData.MiscFeature = testRaw.MiscFeature.astype("category").cat.codes
testCapData.Alley = testRaw.Alley.astype("category").cat.codes
testCapData.Utilities = testRaw.Utilities.astype("category").cat.codes

testCapData.LotFrontage = np.where(pd.isnull(testRaw.LotFrontage), -1, testRaw['LotFrontage'])
testCapData.MasVnrArea = np.where(pd.isnull(testRaw.MasVnrArea), -1, testRaw['MasVnrArea'])
testCapData.GarageYrBlt = np.where(pd.isnull(testRaw.GarageYrBlt), -1, testRaw['GarageYrBlt'])
testCapData.BsmtFinSF1 = np.where(pd.isnull(testRaw.BsmtFinSF1), -1, testRaw['BsmtFinSF1'])
testCapData.BsmtFinSF2 = np.where(pd.isnull(testRaw.BsmtFinSF2), -1, testRaw['BsmtFinSF2'])
testCapData.BsmtUnfSF = np.where(pd.isnull(testRaw.BsmtUnfSF), -1, testRaw['BsmtUnfSF'])
testCapData.TotalBsmtSF = np.where(pd.isnull(testRaw.TotalBsmtSF), -1, testRaw['TotalBsmtSF'])
testCapData.BsmtHalfBath = np.where(pd.isnull(testRaw.BsmtHalfBath), -1, testRaw['BsmtHalfBath'])
testCapData.GarageCars = np.where(pd.isnull(testRaw.GarageCars), -1, testRaw['GarageCars'])
testCapData.GarageArea = np.where(pd.isnull(testRaw.GarageArea), -1, testRaw['GarageArea'])

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras import models
from keras import layers
from keras import initializers
from keras.layers import Dense,Dropout
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn2pmml import PMMLPipeline

from collections import Counter
from keras.layers import CuDNNLSTM, Dense, Bidirectional, Dropout
from keras.layers import Dense, Dropout, BatchNormalization
from numpy.random import seed
from Preprocessing import removeOutliers,bucketing
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor, GradientBoostingClassifier
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MultiLabelBinarizer, LabelBinarizer
from sklearn.kernel_ridge import KernelRidge
import lightgbm as lgb

seed(1)
from tensorflow import set_random_seed
set_random_seed(2)
scaler = MinMaxScaler()
import keras.backend as K

num_cols = 80
num_epochs = 20

trainDf,valDf = train_test_split(trainCapData, test_size=0.333,random_state=42 )

print(trainDf.iloc[:,1:num_cols])
print(pd.DataFrame(trainDf.iloc[:,1:num_cols]).sum)

print(trainDf.max(axis = 0, skipna = True) )
print(trainDf.min(axis = 0, skipna = True) )

xTrain = trainDf.iloc[:,1:num_cols]
xVal = valDf.iloc[:,1:num_cols]
xTest = testCapData.iloc[:,1:num_cols]

testCapData.to_csv(dirName+'testPre.csv', index=False)
print(xTest)
scalerFix = scaler.fit(pd.DataFrame(xTrain).append(pd.DataFrame(xTest)))
xTrain = scalerFix.transform(xTrain)
xVal = scalerFix.transform(xVal)
xTest = scalerFix.transform(xTest)

yTrain = trainDf.iloc[:,num_cols]
yVal = valDf.iloc[:,num_cols]
# yTest = testCapData.iloc[:,num_cols]

KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)

KRR.fit(xTrain, yTrain)

model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)

model_lgb.fit(xTrain, yTrain)

yPred = KRR.predict(xVal)
yPred = model_lgb.predict(xVal)

print(metrics.mean_squared_error(yPred, yVal))
print(metrics.mean_absolute_error(yPred, yVal))

yTest = KRR.predict(xTest)
yTest = model_lgb.predict(xTest)

pd.DataFrame(yTest).to_csv(dirName+'pred.csv', index=False)
