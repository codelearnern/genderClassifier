# Mostly stolen(not all though) from https://pythonprogramming.net/loading-custom-data-deep-learning-python-tensorflow-keras/

import cv2
import numpy as np
import pickle
import os
from tqdm import tqdm
import random

DATA_DIR = "Training"
CATEGORIES = ["male", "female"]
IMG_SIZE = 150

def create_training_data(training_data, categories, data_dir, img_size):
	for category in categories:
		path = os.path.join(data_dir, category)
		class_num = categories.index(category)

		for img in tqdm(os.listdir(path)):
			try:
				img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)  # convert to array
				new_array = cv2.resize(img_array, (img_size, img_size))
				training_data.append([new_array, class_num])

			except Exception as e:
				pass

training_data = []

create_training_data(training_data, CATEGORIES, DATA_DIR, IMG_SIZE)

random.shuffle(training_data) # Making the training_data more random

x = []
y = []

for features, label in training_data:
	x.append(features)
	y.append(label)

x = np.array(x).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

with open("x.pickle", "wb") as f:
	pickle.dump(x, f)

with open("y.pickle", "wb") as f:
	pickle.dump(y, f)