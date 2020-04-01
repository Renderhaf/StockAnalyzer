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

def unEnlarge(data: list, index: int=-1)->None:
    if index==-1:
        for i in range(len(data)):
            data[i] = mapp(data[i], 0, 1, 0.4, 0.6)
    else:
        for i in range(len(data[index])):
            data[index][i] = mapp(data[index][i], 0, 1, 0.4, 0.6)

learner = sdl(activationFunction="sigmoid", hiddenLayersFunction="relu")

def train(epochs = 10):
    with open("LearningData.json", "r") as file:
        data = json.loads(file.read())

    start = 0
    inputdata = np.asarray(data["input"][start:])
    outputdata = np.asarray(data["output"][start:])

    for i in range(len(inputdata)):
        divisor = (inputdata[i][0]*2)
        inputdata[i] /= divisor
        outputdata[i] /= divisor

        enlarge(inputdata, i)
        enlarge(outputdata, i)

    learner.train(inputdata, outputdata, epochs)

def predict(inputData: list) -> list:
    newInputData = np.asarray([inputData])
    divisor = (newInputData[0][0]*2)
    newInputData /= divisor
    enlarge(newInputData)

    predictedOutput = learner.predict(newInputData)
    unEnlarge(predictedOutput[0])
    predictedOutput = predictedOutput[0][0]
    predictedOutput *= divisor

    return predictedOutput


def plotPredictions(inputData, outputData, show=True):
    lastInputs = [x[-1] for x in inputData]

    print("Input len: {}, Output len: {}, first Input: {}, first output: {}".format(len(inputData), len(outputData), inputData[0], outputData[0]))
    predictedOutputs = []
    
    for i in range(len(inputData)):
        predictedOutputs.append(float(predict(inputData[i])))
        # if i < 10:
        #     print("The predicted output is {}, while the true output is {}".format(predictedOutputs[i], outputData[i]))

    match = 0
    count = 0

    for lastVal, predicted, real in zip(lastInputs, predictedOutputs, outputData):
        realDiff = real[0] - lastVal
        preDiff = predicted - lastVal

        if (realDiff > 0 and preDiff > 0)  or (realDiff < 0 and preDiff < 0):
            match += 1
        
        count += 1

    print("Diff match precentage is {}.".format((match/count)*100))

    if show:
        plt.plot(outputData)
        plt.plot(predictedOutputs)
        plt.legend(["Real Outputs", "Predicted Outputs"])
        plt.show()


def predictMax(inputData, outputData):
    predictionData = inputData[0]

    for i in range(len(inputData)): 
        workData = predictionData[i:i+7]
        prediction = float(predict(workData))

        if i < 10:
            print(str(workData) + " ---> " + str(prediction))

        predictionData.append(prediction)

    plt.plot(predictionData)
    plt.show()


def test():
    isNewModel = True
    isTrain = True
    
    if isTrain:
        if isNewModel:
            train(epochs = 10)
            learner.save()
        else:
            learner.load()

    inputData, outputData = sda.splitIntoChunks("GOOG")
    plotPredictions(inputData, outputData, show=True)

if __name__ == "__main__":
    test()
    


    


