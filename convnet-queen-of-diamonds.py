from sre_parse import CATEGORIES
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
import tensorflow

DATADIR = "I:/Dev/queen-of-diamonds-dataset"
CATEGORIES = ["notQd_FULL", "QueenOfDiamonds_FULL"]

training_data = []
IMG_SIZE = 200

def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
            # img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
            # plt.imshow(img_array)
            # plt.show()
            print(os.path.join(path,img))
            # print(img_array.shape) #(480, 640, 3)
            # Resize images
            # IMG_SIZE = 200
            new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
            training_data.append([new_array, class_num])
            # plt.imshow(new_array, cmap="gray")
            # plt.show()

CREATE_TRAINING_DATA = True
if CREATE_TRAINING_DATA:
    create_training_data()
    # np.save('features.npy', training_data) #saving
# training_data = np.load('features.npy') #loading

import random
random.shuffle(training_data)
for sample in training_data[:10]:
    print(sample[1])

x_train = []
y_train = []
x_test = []
y_test = []

for features, label in training_data[:-200]:
    x_train.append(features)
    y_train.append(label)
for features, label in training_data[-200:]:
    x_test.append(features)
    y_test.append(label)

x_train = np.array(x_train).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
x_test = np.array(x_test).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
# y_train = np.asarray(y_train).astype("uint8")
y_train = np.array(y_train)
y_test = np.array(y_test)
print(y_train)
tensorflow.one_hot(y_train, depth=2)
# y_train = np.array(y_train).reshape(-1, 1)
print(y_train)
x_val = np.array(x_train[:1600])
partial_x_train = np.array(x_train[1600:])
y_val = np.array(y_train[:1600])
partial_y_train = np.array(y_train[1600:])
print("partial_x_train: ", partial_x_train)
print("x_train: ", x_train)
print("y_train: ", y_train)
print("partial_y_train: ", partial_y_train)
print("shape(y_train: ", y_train.shape)
print("x_train.shape: ", x_train.shape)
print("type(x_train) :", type(x_train))
print("type(y_train): ", type(y_train))

from tensorflow import keras
from tensorflow.keras import layers
# model = keras.Sequential([
#     layers.Dense(256, activation="relu"),
#     layers.Dense(128, activation="relu"),
#     layers.Dense(1, activation="sigmoid")
# ])

# model.compile(optimizer="rmsprop",
#               loss="binary_crossentropy",
#               metrics=["accuracy"])

# history = model.fit(x_train,
#                     y_train,
#                     epochs=20,
#                     batch_size=10,
#                     validation_split=0.2)

import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D

X = x_train/255.0
y = y_train
model = Sequential()

model.add(Conv2D(64, (3, 3), input_shape=X.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

model.add(Dense(64))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(X, y, batch_size=1, epochs=10, validation_split=0.1)

history_dict = history.history
print(history_dict.keys())

loss_values = history_dict["loss"]
val_loss_values = history_dict["val_loss"]
epochs = range(1, len(loss_values) + 1)
plt.plot(epochs, loss_values, "bo", label="Training loss")
plt.plot(epochs, val_loss_values, "b", label="Validation loss")
plt.title("Training and validation loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

plt.clf()
acc = history_dict["accuracy"]
val_acc = history_dict["val_accuracy"]
plt.plot(epochs, acc, "bo", label="Training acc")
plt.plot(epochs, val_acc, "b", label="Validation acc")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

results = model.evaluate(x_test, y_test)

print("results: ", results)

model.predict(x_test)

import keyboard
index = -200
while True:
    if keyboard.is_pressed("q"):
        break
    if keyboard.is_pressed("o"):
        filename = input("Filename to open: ")
    if keyboard.is_pressed("r"):
        for features, label in training_data[-200:]:
            plt.imshow(features, cmap="gray")
            print("index: ", index)
            index += 1
            print("label: ", label)
            plt.show()
        