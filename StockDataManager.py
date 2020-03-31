import datetime as dt
import pandas_datareader.data as web
import json
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style as pltstyle
import time
import os

dataPath = "./data/"

def saveStockData(stockName: str, yearsBack:float=3):
    currentYear = int(time.asctime().split(" ")[-1])
    df = web.DataReader(stockName, "yahoo", start=dt.datetime(currentYear - yearsBack, 1, 1), end=dt.datetime(currentYear, 1, 1))
    df.to_csv(dataPath + stockName + ".csv")

def test():
    data = pd.read_csv("AAPL.csv")
    closedata = list(data.Close)
    datedata = data.Date

    # print("Data type for plotted data is {}. it looks like this:\n".format(type(closedata)))
    # print(closedata[-100:])

    plt.plot(closedata)
    plt.show()


def saveStocks(stocks: list):
    for stock in stocks:
        saveStockData(stock)

def plotStocks(stocks: list, colors: list =[]):
    if colors == []:
        for stock in stocks:
            data = pd.read_csv(dataPath + stock + ".csv")
            closedata = list(data.Close)
            plt.plot(closedata)
    else:
        for stock, color in zip(stocks, colors):
            data = pd.read_csv(dataPath + stock + ".csv")
            closedata = list(data.Close)
            plt.plot(closedata, color)
    plt.show()

def isSaved(stockName:str)->bool:
    return (stockName+".csv") in os.listdir(dataPath)

def makeDataIntoDict(data:pd.DataFrame)->dict:
    dataDict = dict()
    dataDict["c"] = list(data.Close)
    dataDict["t"] = list(data.Date)
    return dataDict

def getData(stockName: str) -> dict:
    if isSaved(stockName):
        data = pd.read_csv(dataPath+stockName+".csv")    
    else:
        #If its not saved, save it and start again
        saveStockData(stockName)
        return getData(stockName)

    return makeDataIntoDict(data)

if __name__ == "__main__":
    companies = ['AAPL', 'GOOGL', 'TSLA', 'INTC', 'AMZN', 'AMD']
    # colors = ['r', 'g', 'b', 'cyan', 'lime', 'magenta']
    # plotStocks(companies)
    # test()
    saveStockData("GOOG")
