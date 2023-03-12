import pickle

import keras
import keras.layers as layers
import numpy as np
import tensorflow as tf
import torch


def read_model(model_file: str = None):
    if model_file is None:
        return None
    file = open(model_file, "rb")
    weights = pickle.load(file)
    file.close()
    return weights


# Generator for batching the data, as the data is too large to handle in one run
# General idea from https://medium.com/analytics-vidhya/train-keras-model-with-large-dataset-batch-training-6b3099fdf366
def batch_generator(x: np.ndarray, y: np.ndarray, batch_size: int, steps: int):
    index = 0
    while True:
        yield (x[index*batch_size:index*batch_size + batch_size],
               y[index*batch_size:index*batch_size + batch_size])
        index = index + 1 if index < steps else 0


class LSTM:
    def __init__(self, model_file: str = None) -> None:
        self.model = None
        read_model(model_file)

    def train_model(self,
                    train_x: np.ndarray,
                    train_y: np.ndarray,
                    test_x: np.ndarray,
                    test_y: np.ndarray,
                    output_file: str = None,
                    epochs: int = 3,
                    steps_per_epoch: int = 5,
                    batches: int = 1) -> None:
        model = keras.Sequential()
        model.add(layers.Input(shape=(train_x.shape[1], 1,)))
        model.add(layers.LSTM(100))
        model.add(layers.Dense(train_y.shape[1], activation="softmax"))
        model.compile(loss="categorical_crossentropy", metrics=["accuracy"])

        training_generator = batch_generator(train_x, train_y, batches, steps_per_epoch)

        model.fit(train_x, train_y,
                  batch_size=batches,
                            validation_data=(test_x, test_y),
                            epochs=epochs,
                            steps_per_epoch=steps_per_epoch,
                            verbose=1,
                            use_multiprocessing=True)

        # model.fit_generator(training_generator,
        #                     validation_data=(test_x, test_y),
        #                     epochs=epochs,
        #                     steps_per_epoch=steps_per_epoch,
        #                     verbose=1,
        #                     use_multiprocessing=True)

        if output_file is not None:
            file = open(output_file, "wb")
            pickle.dump(model, file)
            file.close()

        self.model = model

    def evaluate(self, test_x: np.ndarray, test_y: np.ndarray) -> float:
        return self.model.evaluate(test_x, test_y)

    def classify(self, audio_data: np.ndarray) -> str:
        self.model.call(audio_data)
