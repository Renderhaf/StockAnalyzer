import keras
import numpy as np

class StockDataLearner:
    def __init__(self, inputSize = 7, outputSize = 1, activationFunction="linear"):
        self.activationFunction = activationFunction
        self.model = keras.Sequential()
        self.model.add(keras.layers.Dense(inputSize))
        self.model.add(keras.layers.Dense(128, activation="relu"))
        self.model.add(keras.layers.Dense(512, activation="relu"))
        self.model.add(keras.layers.Dense(512, activation="relu"))
        self.model.add(keras.layers.Dense(128, activation="relu"))
        self.model.add(keras.layers.Dense(outputSize, activation=activationFunction))

        self.model.compile(optimizer='adam',
                    loss='mse', 
                    metrics=['mse', 'mae'])

    def train(self, trainInput, trainOutput, e_num=3):
        self.model.fit(trainInput, trainOutput, epochs=e_num)

    def predict(self, inputData) -> list:
        prediction = self.model.predict(inputData)
        prediction /= 2 if self.activationFunction == "softmax" else 1
        return prediction

