from config import *
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
import keras
from keras import layers
from keras import Model
from keras.models import Sequential
from tensorflow.keras.utils import to_categorical, plot_model
from keras.layers import Dense, Flatten, Conv3D, MaxPooling3D, Dropout, LSTM, ZeroPadding3D, Input, TimeDistributed, Conv2D, Activation
from keras.applications.vgg16 import VGG16
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv3D, MaxPooling3D, Dropout, LSTM, ZeroPadding3D, Input, TimeDistributed, Conv2D, Activation
from tensorflow.keras.optimizers import SGD
from keras.applications.vgg16 import VGG16      


def create_model(model_type):
    global model
    if model_type == CNN_type:
        create_3D_CNN_model()
        compile_model()
        model.save(code_base_directory + "/Run-time/Models/CNN_3D_Model")
    else:
        create_LSTM_model()
        compile_model()
        model.save(code_base_directory + "Run-time/Models/LSTM_Model")

def compile_model():
    global model
    # Compile the model
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=tf.optimizers.Adam(lr=learning_rate),
                  metrics=['accuracy'])

def load_model(model_type):
    global model
    if model_type == CNN_type:
        model = keras.models.load_model(code_base_directory + "/Run-time/Models/CNN_3D_Model")
    else:
        model = keras.models.load_model(code_base_directory + "/Run-time/Models/LSTM_Model")

def create_3D_CNN_model():
    global model
    model = Sequential(name="3D-CNN Model")

    # 1st layer group
    model.add(Conv3D(64, (3, 3, 3), activation="relu",name="conv1", 
                        input_shape=(D, W, H, C),
                        strides=(1, 1, 1), padding="same"))  
    model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), name="pool1", padding="valid"))

    # 2nd layer group  
    model.add(Conv3D(128, (3, 3, 3), activation="relu",name="conv2", 
                        strides=(1, 1, 1), padding="same"))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), name="pool2", padding="valid"))

    # 3rd layer group   
    model.add(Conv3D(256, (3, 3, 3), activation="relu",name="conv3a", 
                        strides=(1, 1, 1), padding="same"))
    model.add(Conv3D(256, (3, 3, 3), activation="relu",name="conv3b", 
                        strides=(1, 1, 1), padding="same"))	
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), name="pool3", padding="valid"))

    # 4th layer group  
    model.add(Conv3D(512, (3, 3, 3), activation="relu",name="conv4a", 
                        strides=(1, 1, 1), padding="same"))   
    model.add(Conv3D(512, (3, 3, 3), activation="relu",name="conv4b", 
                        strides=(1, 1, 1), padding="same"))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), name="pool4", padding="valid"))

    # 5th layer group  
    model.add(Conv3D(512, (3, 3, 3), activation="relu",name="conv5a", 
                        strides=(1, 1, 1), padding="same"))   
    model.add(Conv3D(512, (3, 3, 3), activation="relu",name="conv5b",
                        strides=(1, 1, 1), padding="same"))
    model.add(ZeroPadding3D(padding=(0, 1, 1)))	
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), name="pool5", padding="valid"))
    model.add(Flatten())
                        
    # FC layers group
    model.add(Dense(4096, activation='relu', name='fc6'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu', name='fc7'))
    model.add(Dropout(.5))
    model.add(Dense(487, activation='softmax', name='fc8'))


    #if weights_path:
    #    model.load_weights(weights_path)

    #Make Changes for our model.
    model.layers.pop() #Remove last layer. as the number of classes used is not as same our data.
    pre_last_layer_output = model.layers[-1].output
    last_layer_output = Dense(no_classes, activation='softmax', name='fc9')(pre_last_layer_output)

    model = Model(model.input, last_layer_output)

# de incercat trainable!

    for layer in model.layers[:-5]:
        layer.trainable=False

# de incercat fara weights
def build_vgg(shape):
    vgg = VGG16(weights="imagenet", input_shape=shape, include_top=False)
    
    return vgg

def create_LSTM_model():
    global model

    vggModel = build_vgg((W,H,C))
    for layer in vggModel.layers[:-1]:
        layer.trainable=False

    model = Sequential()
    input_layer = Input(shape=(D, W, H, C))
    model = TimeDistributed(vggModel)(input_layer) 
    model = TimeDistributed(Flatten())(model)
    
    model = LSTM(128, return_sequences=False)(model)
    model = Dropout(.5)(model)
    
    output_layer = Dense(no_classes, activation='softmax')(model)

    model = Model(input_layer, output_layer)

def train(X_train, y_train, val_split):
    global model
    global validation_split
    validation_split = val_split
    
    # Convert target vectors to categorical targets
    y_train = to_categorical(y_train).astype(np.integer)
    
    # Fit data to model
    history = model.fit(X_train, y_train,
                batch_size=batch_size,
                epochs=no_epochs,
                verbose=verbosity,
                validation_split=val_split)

def test(X_test, y_test):
    # Convert target vectors to categorical targets
    y_test = to_categorical(y_test).astype(np.integer)
    
    # Generate generalization metrics
    model_loss, model_accuracy = model.evaluate(X_test, y_test, verbose=0)
    y_pred = model.predict(X_test)

    return model_loss, model_accuracy, y_pred

def predict(X_test):
        y_pred = model.predict(X_test)
        return y_pred

def plot_model_metrics(history):
    plt.figure(figsize=(8, 4))
    plt.title('Model Performance for DNN Video Processing', size=18, c="C7")
    plt.ylabel('Loss value', size=15, color='C7')
    plt.xlabel('Epoch No.', size=15, color='C7')
    plt.plot(history.history['loss'],  'o-', label='Training Data Loss', linewidth=2, c='C3') #C3 = red color.
    plt.plot(history.history['accuracy'],  'o-', label='Training Data Accuracy', linewidth=2, c='C2') #C2 = green color.

    if len(history.history) > 2:
        plt.plot(history.history['val_accuracy'],  'o-', label='Validation Data Accuracy', linewidth=2, c='b') #b = blue color.

    plt.legend()    
    plt.show()

def show_model_performance():
    global model
    plot_model_metrics(model.history)

def show_summary():
    global model
    print(model.summary())

def save_model_plot(name):
    global model
    plot_model(model, to_file=name+'.png', show_shapes = True)