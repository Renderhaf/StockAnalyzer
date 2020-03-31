import json
import matplotlib.pyplot as plt

with open("LearningData.json", 'r') as file:
    data = json.loads(file.read())

startIndex = 4000
inputs = data['input'][startIndex:]
outputs = data['output'][startIndex:]


def checkdiff():
    length  = 755
    diffs = []
    summ = 0
    amount = 0 

    for inputt, output in zip(inputs, outputs):
        diffs.append(inputt[-1] - output[-1])
        summ += abs(inputt[-1] - output[-1])
        amount += 1
        # if amount % 755 == 0:
        #     plt.plot(diffs)
        #     diffs = []

    print(summ/amount)

    plt.plot(diffs)
    plt.show()

def showContinuous():
    data = []
    for index, input_set in enumerate(inputs):
        for day in input_set:
            data.append(day)
        data.append(outputs[index][0])
    
    print(data[:10])
    plt.plot(data)
    plt.show()

checkdiff()
