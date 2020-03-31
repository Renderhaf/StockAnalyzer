import keras
import numpy as np

class StockDataLearner:
    def __init__(self, inputSize = 7, outputSize = 1, activationFunction="linear", hiddenLayersFunction="relu"):
        self.activationFunction = activationFunction
        self.hiddenLayersFunction = hiddenLayersFunction
        self.model = keras.Sequential()
        self.model.add(keras.layers.Dense(inputSize))
        self.model.add(keras.layers.Dense(128, activation=hiddenLayersFunction))
        self.model.add(keras.layers.Dense(256, activation=hiddenLayersFunction))
        self.model.add(keras.layers.Dense(outputSize, activation=activationFunction))

        self.model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                    loss='mse', 
                    metrics=['mse'])

    def train(self, trainInput, trainOutput, e_num=3):
        self.model.fit(trainInput, trainOutput, epochs=e_num)

    def predict(self, inputData) -> list:
        prediction = self.model.predict(inputData)
        prediction /= 2 if self.activationFunction == "softmax" else 1
        return prediction

