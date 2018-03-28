import os
import csv
import cv2
import numpy as np
import sklearn
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

# read the info from log file
samples = []
with open('./data1/data1/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

# split the data to be traing data and validation data(20%)
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

print('Train samples: {}'.format(len(train_samples)))
print('Validation samples: {}'.format(len(validation_samples)))

# set the steering offset for all three cameras
steering_correction = 0.2
sterring_correction_list = [0., steering_correction, steering_correction*-1.]

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

# model architecture
model = Sequential()

model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape = (160,320,3)))
model.add(Cropping2D(cropping = ((70,25),(0,0))))

model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu'))
model.add(Dropout(0.2))
model.add(Convolution2D(36,5,5, subsample=(2,2),activation='relu'))
model.add(Dropout(0.2))
model.add(Convolution2D(48,5,5, subsample=(2,2),activation='relu'))
model.add(Dropout(0.2))
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(Dropout(0.2))
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(100, activation = 'relu'))
model.add(Dense(50, activation = 'relu'))
model.add(Dense(1))

# training and validating the model
model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch= \
            len(train_samples), validation_data=validation_generator, \
            nb_val_samples=len(validation_samples), nb_epoch=10)
model.save('model.h5')
