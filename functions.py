import tensorflow as tf
import numpy as np
def getData():
    data = list()
    with open("nestor files/mfeat-pix.txt", 'r') as file:
        file = file.readlines()
        for line in file:
            lines = list()
            for character in line:
                if character.isdigit():
                    lines.append(int(character))
            data.append(lines)
    return data


def splitData(data : list):
    counter = 0
    testData = list()
    trainingData = list()
    for line in data:
        if counter < 100:
            trainingData.append(line)
        else:
            testData.append(line)
        counter += 1
        if counter == 200:
            counter = 0
    return testData, trainingData

def createLabelTensor():
    n = list()
    for i in range(1000):
        n.append(int(i/100))
    n = tf.convert_to_tensor(n)
    return n

def initiateNetwork(sizesOfHiddenLayers : tuple):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(15 * 16))

    for LayerSize in sizesOfHiddenLayers:
        model.add(tf.keras.layers.Dense(LayerSize))

    model.add(tf.keras.layers.Dense(10))
    model.compile()
    return model
