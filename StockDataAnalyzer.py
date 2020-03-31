import StockDataManager as sd
import matplotlib.pyplot as plt
import numpy as np
import json
import random

companiesToAnalyze = ['AAPL', 'GOOGL', 'TSLA', 'INTC', 'AMZN',
                     'AMD','MDB', 'GM', 'DELL', 
                     'ABT', 'EBAY', 'HPQ', 'KEYS', 'NFLX',
                     'MMM', 'ADBE', 'A', 'T', 'AXP',
                     'ACN', 'AKAM', 'CDW', 'CSCO', 'FLIR',
                     'IT', 'GPN', 'HPE', 'IBM', 'MA',
                     'ADSK','BKR','BLK','AVGO','ARNC',
                     'BBY','BA','EQIX','ES','HAL',
                     'GPN','HD','LB','MCD']

def splitIntoChunks(stockName, inputRange: int=7, outputRange:int=1):
    data = sd.getData(stockName)
    closeValues = data['c']
    inputVals = []
    outputVals = []
    for location in range(0, len(closeValues)-inputRange-outputRange):
        inputVals.append(closeValues[location:location+inputRange])
        outputVals.append(closeValues[location+inputRange:location+inputRange+outputRange])

    return inputVals, outputVals

def makeLearningData(comapnies: list):
    inputDatas = []
    outputDatas = []

    for company in comapnies:
        inputData, outputData = splitIntoChunks(company)
        inputDatas.extend(inputData)
        outputDatas.extend(outputData)
    
    #Shuffle the data
    indexlist = list(range(len(inputDatas)))
    random.shuffle(indexlist)

    newinput = []
    newoutput = []
    for i in indexlist:
        newinput.append(inputDatas[i])
        newoutput.append(outputDatas[i])

    return inputDatas, outputDatas
    # return newinput, newoutput

def saveLearningData(inputData:list, outputData:list)->None:
    data = {
        "input":inputData,
        "output":outputData
    }
    with open("LearningData.json", "w") as file:
        file.write(json.dumps(data))

def test():
    inputData, outputData = makeLearningData(companiesToAnalyze)
    print(inputData[0], outputData[0])
    saveLearningData(inputData, outputData)

if __name__ == "__main__":
    test()