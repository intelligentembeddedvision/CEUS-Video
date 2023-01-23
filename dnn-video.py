#Ref https://github.com/dinaYriad/Action-Recognition

from tkinter import N
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (10,6)
import numpy as np
import sklearn as sk
import cv2 as cv
import pandas as pd
import os
import ntpath
import pickle
from tqdm import tqdm
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import tensorflow as tf
import keras
from keras import layers
from keras import Model
from keras.models import Sequential
from tensorflow.keras.utils import to_categorical, plot_model
from keras.layers import Dense, Flatten, Conv3D, MaxPooling3D, Dropout, LSTM, ZeroPadding3D, Input, TimeDistributed, Conv2D, Activation
from tensorflow.keras.optimizers import SGD
from keras.applications.vgg16 import VGG16      
from platform import python_version
from data.read_preproc_data import read_train_data_paths, read_test_data_paths, get_test_data_names, know_about_train_data, know_about_test_data, \
     preprocess, save_structure, load_structure
from models.models import create_model, compile_model, load_model, create_3D_CNN_model, build_vgg, create_LSTM_model, train, test, predict, \
    plot_model_metrics, show_model_performance, show_summary, save_model_plot 
from config import * 

print("Python: ", python_version())
print("tf: ", tf.__version__)
print("keras: ", keras.__version__)
print("opencv: ", cv.__version__)
print("sklearn: ", sk.__version__)

print("Classes:", classes)
print("Test data path:", test_path)
print("Train data path:", train_path)



def get_class(value):
    value = get_value(value)
    for class_name in classes:
        if classes[class_name] == value:
            return class_name

def shuffle(X_data, y_data):
    X_data_series = pd.Series(X_data)
    y_data_series = pd.Series(y_data)

    dataFrame = pd.DataFrame()
    dataFrame = pd.concat([X_data_series, y_data_series], axis=1)

    dataArray = np.array(dataFrame)
    np.random.shuffle(dataArray)
    
    return dataArray[:,0], dataArray[:,1]

#A value is either equal to a 'label index' of a 'categorical list' that indicates which label.
#ex:
#if value = 2 or value = [0, 0, 1, 0, 0]:
#return 2
def get_value(value):
    if type(value) == type([]) or type(value) == type(np.array([])):
        return np.argmax(value)
    return value

def create_predictions_csv_file(model_name, predictions):
    pred = []

    data_names = get_test_data_names()
    for i in range(len(predictions)-1):
        pred.append([data_names[i], get_value(predictions[i])])

    np_predictions = np.array(pred)
    np.savetxt(fname=model_name + "_predictions.csv", X=np_predictions, delimiter=",", fmt='%s')

# **HAVE A LOOK AND PLOTS**
def have_a_look(X_data, y_data):
    plt.figure(figsize=(20, 20))
    for n , i in enumerate(list(np.random.randint(0,len(X_data),36))): #Pick random 36 videos
        plt.subplot(6,6,n+1)
        plt.axis('off')

        label = y_data[i] #ex-> label = [0.2, 0.3, 0.2, 0.8, 0.6]
        plt.title(get_class(get_value(label))) #The highest value is 0.8 which is at class no. 4

        first_frame = X_data[i][0] #Pick first frame of this video.
        if C == 1:
            first_frame = first_frame.reshape((W,H))
        plt.imshow((first_frame * 255).astype(np.uint8))    



# **RUN TO LOAD TRACKERS**
load_tracker = {"CNN_3D_Model":"n",
                "LSTM_Model":"n",
                "train_data":"n",
                "test_data":"n",
                }

for file_name in load_tracker:
    user_input = input("Load " + file_name + " y/n? ")
    load_tracker[file_name] = user_input

print('\n', load_tracker)

# **CHOSE WHICH ALGORITHM TO USE**
model_id = int(input("Which model to run? '0' for 3D_CNN and '1' for LSTM? "))

model_name = 'CNN_3D_Model'

if model_id == 1:
    model_name = "LSTM_Model"

print('\nChosen Model: ' + model_name)

# **TRAINING PROCESS**
if load_tracker["train_data"] == "n":
    print("Reading Training Data..")
    trainingData_paths, y_train = read_train_data_paths(classes, train_path)
    print(trainingData_paths)
    trainingData_paths, y_train = shuffle(trainingData_paths, y_train)
    print(trainingData_paths)
    print(y_train)

    # **CHANGE y_train to get rid of NAN**

    for number in y_train:
        if pd.isnull(number): number=0.0

    y_train2 = pd.DataFrame(y_train)
    y_train2 =y_train2.fillna(0)
    print(y_train2)

    y_train2.to_numpy()
    y_train=np. array(y_train2)
    print(y_train)

# **KNOW ABOUT TRAINING DATA**

if load_tracker["train_data"] == "n":
    know_about_train_data(trainingData_paths)

# **TRAIN DATA PREPROCESSING**

if load_tracker["train_data"] == "n":
    X_train, y_train = preprocess(trainingData_paths, y_train)
    save_structure(X_train, "X_train")
    save_structure(y_train, "y_train")
else:
    X_train = load_structure("X_train")
    y_train = load_structure("y_train")
    print("Training data is Loaded!")

# **HAVE A LOOK AT 36 RANDOM SAMPLES**

have_a_look(X_train, y_train)

# **CREATE/LOAD MODEL**

if load_tracker[model_name] == "n":
    create_model(model_id)
    save_model_plot(model_name)
    print("Model is created!")
else:
    load_model(model_id)
    print("Model is loaded!")

show_summary()

# **START TRAINING THE MODEL**
train(X_train, y_train, val_split=0)
show_model_performance()

# **READ TESTING DATA**

if load_tracker["test_data"] == "n":
    print("Reading Testing Data..")
    testingData_paths = read_test_data_paths()

#KNOW ABOUT TESTING DATA

if load_tracker["test_data"] == "n":
    know_about_test_data(testingData_paths)

#TEST DATA PREPROCESSING**

if load_tracker["test_data"] == "n":
    X_test, _ = preprocess(testingData_paths, [])
    save_structure(X_test, "X_test")
else:
    X_test = load_structure("X_test")
    print("Testing data is Loaded")

# **TEST MODELS**

y_predict = predict(X_test)

have_a_look(X_test, y_predict)

create_predictions_csv_file(model_name, y_predict)

