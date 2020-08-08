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


seed(1)
from tensorflow import set_random_seed
set_random_seed(2)
scaler = MinMaxScaler()
import keras.backend as K

num_cols = 785
num_epochs = 20
train_file_name = "/home/jahnavi/Downloads/digit-recognizer/train.csv"
test_file_name = "/home/jahnavi/Downloads/digit-recognizer/test.csv"
model_h5_file_name = "/home/jahnavi/Downloads/digit-recognizer/modelNn.h5"

dataframe = pd.read_csv(train_file_name)
testDf = pd.read_csv(test_file_name)

trainDf,valDf = train_test_split(dataframe, test_size=0.333,random_state=42 )

print(trainDf.iloc[:,1:num_cols])
print(pd.DataFrame(trainDf.iloc[:,1:num_cols]).sum)

print(trainDf.max(axis = 0, skipna = True) )
print(trainDf.min(axis = 0, skipna = True) )

lb = LabelBinarizer()

print(valDf.shape)
xTrain = trainDf.iloc[:, 1:num_cols]
xVal = valDf.iloc[:, 1:num_cols]
yTrain = lb.fit_transform(pd.DataFrame(trainDf.iloc[:, 0]))
yVal = lb.fit_transform(pd.DataFrame(valDf.iloc[:, 0]))

# yTrain = mlb.fit_transform(pd.DataFrame(yTrain))
print(pd.DataFrame(yTrain).shape)
# yVal = mlb.fit_transform(pd.DataFrame(yVal))

print(pd.DataFrame(yTrain).max(axis = 0, skipna = True) )
print(pd.DataFrame(yTrain).min(axis = 0, skipna = True) )


model = models.Sequential()

initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=42)

model.add(layers.Dense(num_cols-1, activation='sigmoid', input_shape=(num_cols-1,)))
model.add(layers.Dense(157, activation='sigmoid'))
# model.add(Dropout(0.5))
# model.add(BatchNormalization())
model.add(layers.Dense(31, activation='sigmoid'))
# model.add(Dropout(0.6))
# model.add(BatchNormalization())
# model.add(layers.Dense(98, activation='sigmoid'))
# model.add(Dropout(0.7))
# model.add(BatchNormalization())
# model.add(layers.Dense(33, activation='sigmoid'))
# model.add(layers.Dense(11, activation='sigmoid'))
model.add(layers.Dense(10, activation='sigmoid'))
# model.add(Dropout(0.6))
# model.add(BatchNormalization())
# model.add(layers.Dense(50, activation='sigmoid'))

# finalLayer = layers.Dense(1, activation='sigmoid')
# model.add(layers.Dense(4, activation='sigmoid'))
model.compile(optimizer='rmsprop',
              # loss='binary_crossentropy',
              loss = 'mean_squared_error',
              metrics=['acc'])

history = model.fit(xTrain,
                    yTrain,
                   epochs=num_epochs,
                    steps_per_epoch=30,
                   batch_size=None,
                    validation_data=(xVal, yVal),
                    validation_steps=40)

# model.save_weights(model_h5_file_name)
print("Saved model to disk")
model.save(model_h5_file_name)
history_dict = history.history
print(history_dict)

res = model.predict(testDf)

print(res.shape)

yTest = lb.inverse_transform(res)
print(yTest)

pd.DataFrame(yTest).to_csv('/home/jahnavi/Downloads/digit-recognizer/yTest.csv')