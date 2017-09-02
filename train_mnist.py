"""
Train an MNIST image recognition model.
"""

import keras
import numpy as np
from keras.callbacks import EarlyStopping
from keras.datasets import mnist
from keras.layers import (Conv2D, Dense, Dropout, Flatten, Input, MaxPooling1D,
                          MaxPooling2D, BatchNormalization)
from keras.models import Model, Sequential
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())

# training data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = np.expand_dims(x_train / np.max(x_train), -1)
x_test = np.expand_dims(x_test / np.max(x_test), -1)
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# our deep model
input_shape = x_train[0].shape
classes = 10
inputs = Input(shape=input_shape)
# Block 1
x = Conv2D(64, (3, 3), activation='relu',
           padding='same', name='block1_conv1')(inputs)
x = Conv2D(64, (3, 3), activation='relu',
           padding='same', name='block1_conv2')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

# Block 2
x = Conv2D(128, (3, 3), activation='relu',
           padding='same', name='block2_conv1')(x)
x = Conv2D(128, (3, 3), activation='relu',
           padding='same', name='block2_conv2')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

# Block 3
x = Conv2D(256, (3, 3), activation='relu',
           padding='same', name='block3_conv1')(x)
x = Conv2D(256, (3, 3), activation='relu',
           padding='same', name='block3_conv2')(x)
x = Conv2D(256, (3, 3), activation='relu',
           padding='same', name='block3_conv3')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

# Classification block
x = Flatten(name='flatten')(x)
x = Dense(512, activation='relu', name='fc1')(x)
x - BatchNormalization()(x)
x = Dense(512, activation='relu', name='fc2')(x)
x - BatchNormalization()(x)
x = Dense(classes, activation='softmax', name='predictions')(x)

model = Model(inputs, x)
print(model.summary())

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
early_stopping = EarlyStopping(monitor='loss', patience=2)
history = model.fit(x_train, y_train,
                    batch_size=64,
                    epochs=64,
                    verbose=1,
                    callbacks=[early_stopping],
                    validation_data=(x_test, y_test))

# save the model in an HDF5 file, built in to keras
model.save('var/data/mnist.h5')
