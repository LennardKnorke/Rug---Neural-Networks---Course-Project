#Libs
import numpy as np
import tensorflow as tf
#From Other Files
from functions import *

#Load and prepare data
data = getData()

training_data, test_data = splitData(data)

target_data = createLabelTensor()

#Set up Network
Size_Hidden_Layers = (16, 16)#taken for no particular reason
Neural_Network = initiateNetwork(Size_Hidden_Layers)

#Train network


#Set the regulation terms

#Set K for k cross validation

