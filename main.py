
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import random

from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf

from tqdm import tqdm

from sklearn.ensemble import RandomForestClassifier



file_path = "inputs/digit-recognizer/"
train_X = pd.read_csv(f"{file_path}train.csv")
test_X = pd.read_csv(f"{file_path}test.csv")

train_Y = train_X['label']#.to_frame().rename(columns={0: 'label'})
train_X = train_X.drop('label', axis=1)


def preprocess(x, y):
    digits = []
    labels = []
    for i in tqdm(range(0, len(x))):
        digit = np.array(x.iloc[i:i + 1, :])
        # digit = digit.reshape(28, 28)
        # label = y[i]
        digits.append(digit.reshape(28, 28))
        labels.append(y[i])

    digits = np.array(digits)
    labels = np.array(labels)
    return digits, labels


digits,labels = preprocess(train_X , train_Y)


''' setup the model '''
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.BatchNormalization(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.BatchNormalization(),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.BatchNormalization(),
    layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
)

early_stopping = keras.callbacks.EarlyStopping(
    patience=10,
    min_delta=0.001,
    restore_best_weights=True,
)

''' train the model '''
history = model.fit(
    digits, labels,
    batch_size=30,
    epochs=10,
    callbacks=[early_stopping]
)

def pre_process_test(df):
    test_digits = []
    for i in tqdm(range(0, len(df))):
        test_digit = np.array(df.iloc[i:i + 1, :])
        test_digit = test_digit.reshape(28, 28)
        test_digits.append(test_digit)

    test_digits = np.array(test_digits)
    return test_digits

test_digits = pre_process_test(test_X)

ypredtest = model.predict(test_digits)
test_pred = []
for i in tqdm(range(0,len(ypredtest))):
    test_pred.append(np.argmax(ypredtest[i]))
pre_df = pd.DataFrame({'ImageId':test_X.index+1,'Label':test_pred})

pre_df.to_csv("outputs/CNN_predictions.csv",index=False)