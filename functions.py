import tensorflow as tf
import numpy as np

def getData() -> list:
    """
    :function: Opens the text file
    :returns: data as list
    """

    data = list()
    with open("nestor files/mfeat-pix.txt", 'r') as file:
        file = file.readlines()
        for line in file:
            lines = list()
            for character in line:
                if character.isdigit():
                    lines.append(int(character)/9.0)#read digit and normalize
            data.append(lines)
    return data


def splitData(data : list) ->tuple(list,list):
    """
    :param: data as a list
    :return: data as tensors tensors
    """

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
    testData = tf.convert_to_tensor(testData)
    trainingData = tf.convert_to_tensor(trainingData)
    return testData, trainingData


def createLabelTensor() ->tf.Tensor:
    """
    returns: tensor with the target values
    """

    n = list()
    for i in range(1000):
        n.append(int(i/100))
    n = tf.convert_to_tensor(n)
    return n


def initiateNetwork(sizesOfHiddenLayers : tuple):
    """
    param: tuple with the sizes of the hidden layers, etc.
    returns: default (feadforward) neural network
    """

    model = tf.keras.Sequential()
    #Starting input layer
    model.add(tf.keras.layers.Dense(15 * 16))

    for LayerSize in sizesOfHiddenLayers:
        model.add(tf.keras.layers.Dense(LayerSize))

    #Numbers 0-9, output layer
    model.add(tf.keras.layers.Dense(10))
    model.compile()
    return model

