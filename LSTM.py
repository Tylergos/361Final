import keras
import keras.layers as layers
import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import pad_sequences


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
        if self.model is None:
            model = keras.Sequential()
            model.add(layers.Input(shape=(train_x.shape[1], 1,)))
            model.add(layers.Conv1D(filters=256, kernel_size=16, dilation_rate=8, padding='same', activation='relu'))
            model.add(layers.MaxPooling1D(pool_size=2))
            model.add(layers.LSTM(528))
            model.add(layers.Dense(train_y.shape[1], activation="softmax"))
            model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=["accuracy"])
            self.model = model

        early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=2)
        if output_file is not None:
            model_checkpoint = ModelCheckpoint(output_file, monitor='val_loss', mode='min', save_best_only=True)

        self.model.fit(train_x, train_y,
                       batch_size=batch_size,
                       validation_data=(test_x, test_y),
                       validation_batch_size=batch_size,
                       epochs=epochs,
                       steps_per_epoch=steps_per_epoch,
                       verbose=1,
                       use_multiprocessing=True,
                       callbacks=[early_stopping, model_checkpoint])

    def evaluate(self, test_x: np.ndarray, test_y: np.ndarray, batch_size: int = 8) -> float:
        return self.model.evaluate(test_x, test_y, batch_size=batch_size)

    def predict(self, audio_data: np.ndarray, batch_size: int = 8) -> np.ndarray:
        return self.model.predict(audio_data, batch_size=batch_size)

    def __convert_long_audio__(self, audio_data: np.ndarray) -> np.ndarray:
        clip_length = self.model.input_shape[1]
        x = []
        for long_clip in audio_data:
            sequences = []
            i = 0
            while i < len(long_clip):
                sequences.append(long_clip[i:i + clip_length])
                i += clip_length
            # We need to pad the clips as they are not all exactly the same length
            sequences = pad_sequences(sequences, maxlen=clip_length, dtype=float)
            x.append(sequences)

        return np.array(x)

    def predict_long(self, audio_data: np.ndarray, batch_size: int = 10):
        shortened_clips = self.__convert_long_audio__(audio_data)
        predictions_averaged = []
        i = 0
        while i < audio_data.shape[0]:
            predictions = self.model.predict(shortened_clips[i], batch_size=batch_size, verbose=1)
            predictions_averaged.append(np.average(predictions, axis=0))
            i += 1
        return np.array(predictions_averaged)
