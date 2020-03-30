import StockDataAnalyzer as sda
from StockDataLearner import StockDataLearner as sdl
import StockDataManager as sdm
import json
import numpy as np
import matplotlib.pyplot as plt

#Some helper functions
def mapp(value:float,  istart:float,  istop:float,  ostart:float,  ostop:float) -> float:
    return ostart + (ostop - ostart) * ((value - istart) / (istop - istart))

def enlarge(data: list, index: int=-1)->None:
    if index==-1:
        for i in range(len(data)):
            data[i] = mapp(data[i], 0.4, 0.6, 0, 1)
    else:
        for i in range(len(data[index])):
            data[index][i] = mapp(data[index][i], 0.4, 0.6, 0, 1)


learner = sdl(activationFunction="softmax")

def train():
    with open("LearningData.json", "r") as file:
        data = json.loads(file.read())

    inputdata = np.asarray(data["input"])
    outputdata = np.asarray(data["output"])

    for i in range(len(inputdata)):
        divisor = (inputdata[i][0]*2)
        inputdata[i] /= divisor
        outputdata[i] /= divisor

        enlarge(inputdata, i)
        enlarge(outputdata, i)

    learner.train(inputdata, outputdata, 10)

def predict(inputData: list) -> list:
    newInputData = np.asarray([inputData])
    divisor = (newInputData[0][0]*2)
    newInputData /= divisor
    enlarge(newInputData)

    predictedOutput = learner.predict(newInputData)
    predictedOutput[0][0] *= divisor
    return predictedOutput[0][0]

if __name__ == "__main__":
    train()

    data = sdm.getData("GOOG")
    closeValues = data["c"]
    inputRange = 7
    outputRange = 1
    inputData = []
    outputData = []

    for location in range(0, len(closeValues)-inputRange-outputRange):
        inputData.append(closeValues[location:location+inputRange])
        outputData.extend(closeValues[location+inputRange:location+inputRange+outputRange])

    inputData = np.asarray(inputData)
    outputData = np.asarray(outputData)
    
    print("Input len: {}, Output len: {}, first Input: {}, first output: {}".format(len(inputData), len(outputData), inputData[0], outputData[0]))
    predictedOutputs = []
    
    for i in range(len(inputData)):
        predictedOutputs.append(float(predict(inputData[i])))
        if i < 5:
            print("The predicted output is {}, while the true output is {}".format(predictedOutputs[i], outputData[i]))

    plt.plot(outputData)
    plt.plot(predictedOutputs)
    plt.legend(["Real Outputs", "Predicted Outputs"])
    plt.show()


    


