import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Training the model
dataset_train = pd.read_excel("DADOS CLIMÁTICOS APP-2.xlsx")
training_set = dataset_train.iloc[:,1:2].values

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
training_set_scaled = sc.fit_transform(training_set)

x_train = []
y_train = []
for i in range(12, 669):
    x_train.append(training_set_scaled[i-12:i,0])
    y_train.append(training_set_scaled[i,0])
x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.callbacks import ModelCheckpoint

data_path = "C:\\Users\\renat\\OneDrive\\Área de Trabalho\\NASA"

regressor = Sequential()
regressor.add(LSTM(units = 80, return_sequences = True, input_shape = (x_train.shape[1], 1)))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 80, return_sequences = True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 80, return_sequences = True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 80, return_sequences = False))
regressor.add(Dropout(0.2))
regressor.add(Dense(units = 1))
regressor.compile(optimizer = "adam", loss = "mean_squared_error")

#print(regressor.summary())
checkpointer = ModelCheckpoint(filepath=data_path + "\\regressor-{epoch:02d}.hdf5", verbose=1)
#regressor.fit(x_train, y_train, epochs = 20, batch_size = 32)
regressor.fit(x_train, y_train, epochs = 50, batch_size = 32, callbacks=[checkpointer])
regressor.save(data_path + "final_regressor.hdf5")

#regressor = regressor.load(data_path + "\\regressor-100.hdf5")
#regressor = load_model(data_path + "\model-100.hdf5")
regressor.load_weights(data_path + "\\regressor-28.hdf5")

# Testing the model
dataset_test = pd.read_excel("DADOS CLIMÁTICOS APP-2 test.xlsx")
real_precipitation_value = dataset_test.iloc[:,1:2].values

dataset_total = pd.concat((dataset_train["chuva (mm)"], dataset_test["chuva (mm)"]), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 12:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
x_test = []
for i in range(12, 13):
    x_test.append(inputs[i-12:i,0])
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predicted_precipitation_value = regressor.predict(x_test)
predicted_precipitation_value = sc.inverse_transform(predicted_precipitation_value)
