from sre_parse import CATEGORIES
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
import tensorflow

DATADIR = "I:/Dev/queen-of-diamonds-dataset"
CATEGORIES = ["notQd", "QueenOfDiamonds"]

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
for features, label in training_data:
    x_train.append(features)
    y_train.append(label)
x_train = np.array(x_train).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
# y_train = np.asarray(y_train).astype("uint8")
y_train = np.array(y_train)
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
model = keras.Sequential([
    layers.Dense(256, activation="relu"),
    layers.Dense(128, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])

model.compile(optimizer="rmsprop",
              loss="binary_crossentropy",
              metrics=["accuracy"])

history = model.fit(x_train,
                    y_train,
                    epochs=20,
                    batch_size=10,
                    validation_split=0.2)

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