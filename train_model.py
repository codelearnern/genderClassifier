# Model from Pythonprogramming.net(I will probably use Xception later)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
import pickle
import numpy as np

with open("x.pickle", "rb") as f:
	x_train = pickle.load(f)

with open("y.pickle", "rb") as f:
	y_train = pickle.load(f)

x_train = x_train/255.0
y_train = np.array(y_train)

model = Sequential()

model.add(Conv2D(256, (3, 3), input_shape=x_train.shape[1:], activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

model.add(Dense(64))

model.add(Dense(1, activation="sigmoid"))

model.compile(optimizer="Adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(x_train, y_train, epochs=15, batch_size=64)

model.save("model.h5")