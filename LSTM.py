import pickle

import keras
import keras.layers as layers
import numpy as np
import tensorflow as tf


def read_model(model_file: str = None):
    if model_file is None:
        return None
    file = open(model_file, "rb")
    weights = pickle.load(file)
    file.close()
    return weights


class LSTM:
    def __init__(self, model_file: str = None) -> None:
        self.model = None
        read_model(model_file)

    def train_model(self, train_x: tf.Tensor, train_y: tf.Tensor, test_x: tf.Tensor, test_y: tf.Tensor, output_file: str = None) -> None:
        # classifications = np.unique(train_y.numpy())

        model = keras.Sequential()
        model.add(layers.Input(shape=(661504, 1,)))
        model.add(layers.LSTM(100))
        model.add(layers.Dense(50, activation="softmax"))
        # model.add(layers.Dense(np.shape(classifications)[0], activation="softmax"))
        model.compile(loss="sparse_categorical_crossentropy", metrics=["accuracy"])

        model.fit(train_x, train_y, validation_data=(test_x, test_y), epochs=3, steps_per_epoch=5)

        if output_file is not None:
            file = open(output_file, "wb")
            pickle.dump(model, file)
            file.close()

        self.model = model

    def evaluate(self, test_x: np.ndarray, test_y: np.ndarray) -> float:
        return self.model.evaluate(test_x, test_y)

    def classify(self, audio_data: np.ndarray) -> str:
        self.model.call(audio_data)
