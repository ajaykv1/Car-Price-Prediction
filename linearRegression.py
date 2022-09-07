#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 19:55:51 2022

@author: ajaykrishnavajjala
"""
#%%
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
import tensorflow as tf

#%%
data = pd.read_csv("/Users/ajaykrishnavajjala/Documents/School/PHD/Recommender Systems/Tensorflow Course/Linear Regression/cars.csv")
#%%
print(data.head())
data = data.drop("Car", axis=1)
data = data.drop("Model", axis=1)
print(data.head())
#%%
Y = data["CO2"]
X = data.drop("CO2", axis=1)
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=42)
#%%
N,D = x_train.shape
print(N,D)
#%%
linear_regression_model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=x_train.shape[1:]),
        tf.keras.layers.Dense(1)
    ])
#%%
linear_regression_model.compile(
        optimizer=tf.keras.optimizers.Adam(0.01),
        loss='mse'
    )
#%%
result = linear_regression_model.fit(x_train,y_train, validation_data=(x_test,y_test), epochs=100)
#%%
import matplotlib
import matplotlib.pyplot as plt

plt.plot(result.history["loss"], label="loss")
plt.plot(result.history["val_loss"], label="val_loss")
plt.legend()
#%%
linear_regression_model.evaluate(x_test,y_test)