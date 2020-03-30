import StockDataManager as sd
import matplotlib.pyplot as plt
import numpy as np
import json

companiesToAnalyze = ['AAPL', 'GOOGL', 'TSLA', 'INTC', 'AMZN',
                     'AMD', 'BTC-USD', 'MDB', 'GM', 'DELL', 
                     'ABT', 'EBAY', 'HPQ', 'KEYS', 'NFLX',
                     'MMM', 'ADBE', 'A', 'T', 'AXP',
                     'ACN', 'AKAM', 'CDW', 'CSCO', 'FLIR',
                     'IT', 'GPN', 'HPE', 'IBM', 'MA']

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
    
    return inputDatas, outputDatas

def saveLearningData(inputData:list, outputData:list)->None:
    data = {
        "input":inputData,
        "output":outputData
    }
    with open("LearningData.json", "w") as file:
        file.write(json.dumps(data))

def test():
    inputData, outputData = makeLearningData(companiesToAnalyze)
    print(len(inputData), len(outputData))
    saveLearningData(inputData, outputData)

if __name__ == "__main__":
    test()
