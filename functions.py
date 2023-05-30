import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def getData() -> np.array:
    """
    :function: Opens the text file
    :returns: data as numpy array
    """
    data = np.ndarray([2000, 16, 15])
    with open("nestor files/mfeat-pix.txt", 'r') as file:
        file = file.readlines()
        i = 0
        for line in file: 
            k = 0 
            l = 0
            for character in line:
                if character.isdigit():
                    data[i][k][l] = int(character)/6.0#read digit and normalize
                    l+=1
                    if l == 15:
                        l = 0
                        k+=1
            i+=1
    return data


def splitData(data : np.array, test_training_ratio : float) ->tuple:
    """
    :param: data as a list
    :return: data as tensors tensors
    """
    if test_training_ratio < 0.0 or test_training_ratio > 1.0:
        raise ValueError ("Enter a float x between: 0.0 <= x <= 1.0")
    test_size = int(2000 * test_training_ratio)


    testData = np.ndarray((test_size, 16, 15)) 
    test_idx = 0
    trainingData = np.ndarray((2000 - test_size, 16, 15))
    training_idx = 0
    #For every 10 number (0-9)
    for i in range(10):
        #There are 200 entries
        for k in range(200):
            idx = k + (i * 200)
            #If k over a limit, copy into training
            if k < (test_training_ratio * 200):
                trainingData[training_idx] = data[idx]
                training_idx += 1
            #Else copy into testing
            else:
                testData[test_idx]= data[idx]
                test_idx += 1
    return (trainingData, testData)

def drawDigit(data : np.ndarray) ->None:
    plt.imshow(data, cmap='gray')
    return
            

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

