import pickle

import keras
import keras.layers as layers
import numpy as np
import tensorflow as tf
import torch
from keras.callbacks import EarlyStopping, ModelCheckpoint


def read_model(model_file: str = None):
    if model_file is None:
        return None
    return keras.models.load_model(model_file)


class LSTM:
    def __init__(self, model_file: str = None) -> None:
        self.model = read_model(model_file)

    def train_model(self,
                    train_x: np.ndarray,
                    train_y: np.ndarray,
                    test_x: np.ndarray,
                    test_y: np.ndarray,
                    output_file: str = None,
                    epochs: int = 1,
                    steps_per_epoch: int = 5,
                    batch_size: int = 1) -> None:
        model = keras.Sequential()
        model.add(layers.Input(shape=(train_x.shape[1], 1,)))
        model.add(layers.Conv1D(filters=256, kernel_size=16, dilation_rate=8, padding='same', activation='relu'))
        model.add(layers.MaxPooling1D(pool_size=2))
        model.add(layers.LSTM(528))
        model.add(layers.Dense(train_y.shape[1], activation="softmax"))
        model.compile(optimizer='adam',  loss="categorical_crossentropy", metrics=["accuracy"])

        early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=2)
        if output_file is not None:
            model_checkpoint = ModelCheckpoint(output_file, monitor='val_loss', mode='min', save_best_only=True)

        model.fit(train_x, train_y,
                  batch_size=batch_size,
                  validation_data=(test_x, test_y),
                  validation_batch_size=batch_size,
                  epochs=epochs,
                  steps_per_epoch=steps_per_epoch,
                  verbose=1,
                  use_multiprocessing=True,
                  callbacks=[early_stopping, model_checkpoint])

        self.model = model

    def evaluate(self, test_x: np.ndarray, test_y: np.ndarray, batch_size: int = 8) -> float:
        return self.model.evaluate(test_x, test_y, batch_size=batch_size)

    def classify(self, audio_data: np.ndarray) -> str:
        self.model.call(audio_data)
