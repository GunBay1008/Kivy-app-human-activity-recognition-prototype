from importlib.resources import read_binary
import pandas as pd
import numpy as np
import tensorflow as tf
#from keras.utils.vis_utils import plot_model
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers.core import Dense, Dropout
from keras.layers import BatchNormalization
from keras.regularizers import L1L2
from keras import optimizers
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
#import seaborn as sns
import os
import csv

# Activities are the class labels
# It is a 3 class classification
ACTIVITIES = {
    0: 'sitting',
    1: 'walking',
    2: 'runing',
}

def confusion_matrix(Y_true, Y_pred):
    
    Y_true = pd.Series([ACTIVITIES[y] for y in np.argmax(Y_true, axis=1)])
    Y_pred = pd.Series([ACTIVITIES[y] for y in np.argmax(Y_pred, axis=1)])

    return pd.crosstab(Y_true, Y_pred, rownames=['True'], colnames=['Pred'])
    plt.show()  

def read_csv(path,type,size):    
    values = []
    # read the csv-file line by line
    with open(path, 'r') as csvfile:
        data = csv.reader(csvfile, delimiter=',')
        for row in data:
            values.append(row)



    # ignore the header
    values = values[1:]
    # remove timestamp
    temp = []
    for element in values:
        if len(element) > 3:
            temp.append(element[1:4])
    #print(temp, len(temp))
    values = temp
    temp = []
    temp2 = []
    temp_line = []
#read_csv('data/run/Running.csv',None,None)
    # from string to float
    for element in values:
        for items in element:
            temp_line.append(float(items))
        temp.append(temp_line)
        temp_line =[]
        if (len(temp) == size):
            temp2.append(temp)
            temp=[]
    #print(temp)

    X = np.array(temp2).astype(float)
    Y = np.array([type] * len(X))

    return X,Y

# read data from a directory
def read_data(path,size):
    print("reading")
    directorys = []
    files = []
    X = None
    Y = None

    #sit

    for r, d, f in os.walk(f'{path}/sit'):
        for file in f:
            files.append(file)
            
    for f in files:

        print(f'reading {f}')
        print(f'shape of {f} = ' , end = '')
        print(np.shape(read_csv(f'{path}/sit/{f}',0,size)[0]))
        if X is None:
            X = read_csv(f'{path}/sit/{f}',0,size)[0]
            Y = read_csv(f'{path}/sit/{f}',0,size)[1]
        else:
            X = np.append(X,read_csv(f'{path}/sit/{f}',0,size)[0], axis=0)
            Y = np.append(Y,read_csv(f'{path}/sit/{f}',0,size)[1], axis=0)
        print("shape of X = " , end = '')
        print(np.shape(X))
        print("shape of Y = " , end = '')
        print(np.shape(Y))
    #walk
    files = []
    for r, d, f in os.walk(f'{path}/walk'):
        for file in f:
            files.append( file)
            
    for f in files:
        print(f'reading {f}')
        print(f'shape of {f} = ' , end = '')
        print(np.shape(read_csv(f'{path}/walk/{f}',1,size)[0]))
        X = np.append(X,read_csv(f'{path}/walk/{f}',1,size)[0], axis=0)
        Y = np.append(Y,read_csv(f'{path}/walk/{f}',1,size)[1], axis=0)
        print("shape of X = " , end = '')
        print(np.shape(X))
        print("shape of Y = " , end = '')
        print(np.shape(Y))

    #run
    files = []
    for r, d, f in os.walk(f'{path}/run'):
        for file in f:
            files.append( file)
    
    for f in files:
        print(f'reading {f}')
        print(f'shape of {f} = ' , end = '')
        print(np.shape(read_csv(f'{path}/run/{f}',2,size)[0]))
        X = np.append(X,read_csv(f'{path}/run/{f}',2,size)[0], axis=0)
        Y = np.append(Y,read_csv(f'{path}/run/{f}',2,size)[1], axis=0)
        print("shape of X = " , end = '')
        print(np.shape(X))
        print("shape of Y = " , end = '')
        print(Y)
        print(np.shape(Y))
    
    return X,Y
    

X, Y = read_data('data',5)

print(X,Y)

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def NormalizeClasses(data):
    data = [[i]for i in data]
    enc = OneHotEncoder()
    data = enc.fit_transform(data).toarray()
    return data

# X = NormalizeData(X)
Y = NormalizeClasses(Y)

#function to count the number of classes

def count_classes(y):
    buffer = []
    return len(y[0])

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

epochs = 3
batch_size = 5
n_hidden = 32

timesteps = len(X_train[0])
input_dim = len(X_train[0][0])
n_classes = count_classes(y_train)


print(n_classes)
print(timesteps)
print(input_dim)
print(len(X_train))

# Initiliazing the sequential model
model = Sequential()
# Configuring the parameters
model.add(LSTM(n_hidden, input_shape=(timesteps, input_dim)))
# Adding a dropout layer
model.add(Dropout(0.5))
# Adding a dense output layer with sigmoid activation
model.add(Dense(n_classes, activation='sigmoid'))
model.summary()


# Compiling the model
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])


# Training the model
model.fit(X_train, y_train, batch_size=batch_size, validation_data=(X_test, y_test),epochs=epochs)

# Confusion Matrix
confusion_matrix(y_test, model.predict(X_test))

model.save('/Users/vance/Desktop/LSTM/testmodel/')
model.save('test_model.h5')

#tf.keras.utils.plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)


# from sklearn.model_selection import train_test_split

# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# epochs = 100
# batch_size = 5
# n_hidden = 64

# def count_classes(l):
#     map(str, l[0])
#     return len(l)

# timesteps = len(X_train[0])
# input_dim = len(X_train[0][0])
# n_classes = 3


# print(n_classes)
# print(timesteps)
# print(input_dim)
# print(len(X_train))

# # Initiliazing the sequential model
# model = Sequential()
# # Configuring the parameters
# model.add(LSTM(n_hidden, input_shape=(timesteps, input_dim)))
# # Adding a dropout layer
# model.add(Dropout(0.5))
# # Adding a dense output layer with relu activation
# model.add(Dense(n_classes, activation='sigmoid'))
# model.summary()

# # Compiling the model
# #optimizer = tf.optimizers.Adam(learning_rate=0.1)
# model.compile(loss='categorical_crossentropy', optimizer = 'rmsprop', metrics=['accuracy'])

# # Training the model
# model.fit(X_train, y_train, batch_size=batch_size, validation_data=(X_test, y_test),epochs=epochs)

# # Confusion Matrix
# confusion_matrix(y_test, model.predict(X_test))