import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
print(tf.__version__)
from PIL import Image
import os
import random
os.chdir('D:/2T Washer')
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import RMSprop
import matplotlib.pyplot as plt

train=ImageDataGenerator(rescale=1/255)
validation=ImageDataGenerator(rescale=1/255)

train_dataset=train.flow_from_directory(r'D:/2T Washer/train1/',target_size=(200,200),batch_size=3,class_mode='binary')
validation_dataset=validation.flow_from_directory(r'D:/2T Washer/train1/',target_size=(200,200),batch_size=3,class_mode='binary')

train_dataset.class_indices

train_dataset.classes

model = tf.keras.models.Sequential([tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(200, 200, 3)),
                                    tf.keras.layers.MaxPool2D(2, 2),

                                    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
                                    tf.keras.layers.MaxPool2D(2, 2),

                                    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
                                    tf.keras.layers.MaxPool2D(2, 2),

                                    tf.keras.layers.Flatten(),

                                    tf.keras.layers.Dense(512, activation='relu'),

                                    tf.keras.layers.Dense(1, activation='sigmoid')])

model.compile(loss='binary_crossentropy',
             optimizer=RMSprop(learning_rate=0.001),
             metrics=['accuracy'])

model_fit=model.fit(train_dataset, epochs=30, validation_data=validation_dataset)
#model_fit=model.fit(train_dataset, steps_per_epoch=3, epochs=30, validation_data=validation_dataset)

dir_path = 'D:/2T Washer/test'
count = 0
for i in os.listdir(dir_path):
    count = count + 1
    img = image.load_img(dir_path + '/' + i, target_size=(200, 200, 3))
    plt.imshow(img)
    plt.show()

    X = image.img_to_array(img)
    X = np.expand_dims(X, axis=0)
    images = np.vstack([X])
    val = model.predict(images)

    if val == 0:
        print(str(count) + ' item is NG')
        print("\n\n\n\n")
    else:
        print(str(count) + ' item is OK')
        print("\n\n\n\n")
