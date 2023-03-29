import random
import sys

import librosa
import os

import numpy
import numpy as np
import pickle
import speech_recognition as sr
import nltk
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import tensorflow as tf

from Lexicon import Lexicon
from Similarity import Similarity
from LSTM import LSTM

from keras.utils import pad_sequences

DATASET_PATH = "./SpeakersDataset/50_speakers_audio_data"
PICKLE_AUDIO_FILE_PATH = "./AudioData.txt"
PICKLE_TEXT_FILE_PATH = "./TextData.txt"
MODEL_PATH = "model.h5"
TRAIN_SET_PATH = "./TrainSet.txt"
SAMPLE_RATE = 11025 // 8
CLIP_LENGTH = 661504 // 8
SEQUENCES_PER_CLIP = 10
SEQUENCE_LENGTH = CLIP_LENGTH // SEQUENCES_PER_CLIP
BATCH_SIZE = 8
EPOCHS = 120
RANDOM_STATE = 1234


def read_and_save_audio(file_name: str = PICKLE_AUDIO_FILE_PATH) -> None:
    speaker_files = dict()
    for speaker in os.listdir(DATASET_PATH):
        audio_files = list()
        for audio_file in os.listdir(DATASET_PATH + "/" + speaker):
            print(audio_file)
            # open and resample the files from ~22Khz to ~11KHz.
            try:
                audio_files.append(librosa.load(DATASET_PATH + "/" + speaker + "/" + audio_file, sr=SAMPLE_RATE)[0])
            except:
                print("File failed to resample: " + audio_file)
        speaker_files[speaker] = audio_files
    pickle_file = open(file_name, "wb")
    pickle.dump(speaker_files, pickle_file)
    pickle_file.close()


def read_saved_audio(file_name: str = PICKLE_AUDIO_FILE_PATH) -> dict[list[str]]:
    file = open(file_name, "rb")
    audio_data = pickle.load(file)
    file.close()
    return audio_data


def save_train_set(train_set = list[list[int]], file_name: str = TRAIN_SET_PATH) -> None:
    pickle_file = open(file_name, "wb")
    pickle.dump(train_set, pickle_file)
    pickle_file.close()


def read_train_set(file_name: str = TRAIN_SET_PATH) -> list[list[int]]:
    file = open(file_name, "rb")
    train_set = pickle.load(file)
    file.close()
    return train_set


def train_validation_split(data: dict[list[any]]) -> dict[list[any]]:
    random.seed(RANDOM_STATE)
    train_indexes = {author: random.sample([x for x in range(len(val))], int(len(val) * 0.8)) for (author, val) in
                     data.items()}

    train_data = {}
    validation_data = {}
    for (author, current_data) in data.items():
        train_data[author] = [current_data[i] for i in train_indexes[author]]
        validation_data[author] = [current_data[i] for i in
                                         set.difference(set(range(len(current_data))), set(train_indexes[author]))]
    return train_data, validation_data


def speech_to_text(file_name: str = PICKLE_TEXT_FILE_PATH, quick_run: bool = False) -> None:
    r = sr.Recognizer()
    speaker_files = dict()
    for speaker in os.listdir(DATASET_PATH):
        text_files = list()
        for audio_file in os.listdir(DATASET_PATH + "/" + speaker):
            print(audio_file)

            try:
                with sr.AudioFile(DATASET_PATH + "/" + speaker + "/" + audio_file) as source:
                    audio = r.record(source)
                text_files.append(r.recognize_google(audio))

                if quick_run:
                    break
            except LookupError:  # speech is unintelligible
                print("Could not understand audio")
            except:  # file is corrupted or has issues
                print("File is corrupted or has issues")
        speaker_files[speaker] = text_files
        pickle_file = open(file_name, "wb")
        pickle.dump(speaker_files, pickle_file)
        pickle_file.close()


def read_saved_text(file_name: str = PICKLE_TEXT_FILE_PATH) -> dict[list[str]]:
    file = open(file_name, "rb")
    text_data = pickle.load(file)
    file.close()
    return text_data


def convert_audio_data(audio_data: dict[list[float]], clip_length: int = CLIP_LENGTH) -> tuple[np.ndarray, np.ndarray]:
    x = []
    y = []
    for (author, data) in audio_data.items():
        for clip in data:
            # don't need the sample rate, thus we only take index 0
            x.append(clip)
            y.append(int(author[-2::]))

    y = np.array(y)
    y = tf.keras.utils.to_categorical(y, num_classes=len(numpy.unique(y)))

    # We need to pad the clips as they are not all exactly the same length
    x = pad_sequences(x, maxlen=clip_length, dtype=float)
    return x, y


def split_sequence(audio_data: dict[list[float]], sequence_length: int, keep_partial_splits: bool = False) -> dict[list[float]]:
    sequence_data = dict()
    for (author, data) in audio_data.items():
        sequences = list()
        for clip in data:
            i = 0
            while i + (sequence_length if not keep_partial_splits else 0) < len(clip):
                sequences.append(clip[i:i + sequence_length])
                i += sequence_length
        sequence_data[author] = pad_sequences(sequences, maxlen=sequence_length, dtype=float)

    return sequence_data


def extend_sequence(audio_data: dict[list[float]], sequence_length: int) -> dict[list[float]]:
    combined_audio_data = dict()
    for (author, data) in audio_data.items():
        combined_audio_data[author] = [np.hstack(data)]
    return split_sequence(combined_audio_data, sequence_length, keep_partial_splits=True)


def convert_text_data(text_data: dict[list[str]]) -> tuple[np.ndarray, np.ndarray]:
    x = []
    y = []
    for (author, data) in text_data.items():
        for document in data:
            # don't need the sample rate, thus we only take index 0
            x.append(document)
            y.append(int(author[-2::]))

    return np.array(x), np.array(y)


def train_and_run_LSTM_model(train_data: dict[list[str]], model_file: str = MODEL_PATH, continue_training: bool = False) -> None:
    data = split_sequence(train_data, SEQUENCE_LENGTH)
    x, y = convert_audio_data(data, SEQUENCE_LENGTH)

    train_x, test_x, train_y, test_y = train_test_split(x, y, train_size=0.8, random_state=RANDOM_STATE)

    lstm_model = LSTM(model_file if continue_training else None)
    lstm_model.train_model(train_x, train_y, test_x, test_y, steps_per_epoch=len(train_y) // BATCH_SIZE,
                           batch_size=BATCH_SIZE, output_file=model_file, epochs=EPOCHS)
    lstm_model.evaluate(test_x, test_y)


def train_and_run_LSTM_model_kfold(train_data: dict[list[str]]) -> None:
    data = split_sequence(train_data, SEQUENCE_LENGTH)
    x, y = convert_audio_data(data, SEQUENCE_LENGTH)

    kfold = StratifiedKFold(5)

    accuracies = []
    for fold, (train_index, test_index) in enumerate(kfold.split(x, y)):
        train_x = x[train_index]
        train_y = y[train_index]
        test_x = x[test_index]
        test_y = y[test_index]

        lstm_model = LSTM()
        lstm_model.train_model(train_x, train_y, test_x, test_y, steps_per_epoch=len(train_y) // BATCH_SIZE,
                               batch_size=BATCH_SIZE)
        accuracies.append(lstm_model.evaluate(test_x, test_y))
    print(accuracies)


def train_and_run_lexicon_model_kfold(text_data: dict[list[str]]) -> None:
    x, y = convert_text_data(text_data)

    kfold = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)

    accuracies = []
    for fold, (train_index, test_index) in enumerate(kfold.split(x, y)):
        train_x = x[train_index]
        train_y = y[train_index]
        test_x = x[test_index]
        test_y = y[test_index]

        lexicon_model = Lexicon(train_x, train_y)
        accuracy, _ = lexicon_model.evaluate(test_x, test_y)
        accuracies.append(accuracy)
        print("Fold Done")

    print(np.sum(accuracies) / 5)


def train_and_run_similarity_model_kfold(text_data: dict[list[str]]) -> None:
    x, y = convert_text_data(text_data)

    kfold = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)

    accuracies = []

    for fold, (train_index, test_index) in enumerate(kfold.split(x, y)):
        train_x: np.ndarray = x[train_index]
        train_y: np.ndarray = y[train_index]
        test_x: np.ndarray = x[test_index]
        test_y: np.ndarray = y[test_index]
        similarity_model = Similarity(train_x, train_y)
        accuracy, _ = similarity_model.evaluate(test_x, test_y)
        accuracies.append(accuracy)

        print("Fold Done")

    print(np.sum(accuracies) / 5)


def print_statistics(y_true, y_pred):
    y_pred = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_true, axis=1)

    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred, average="macro")
    precision = precision_score(y_true, y_pred, average="macro")
    f1 = f1_score(y_true, y_pred, average="macro")
    print(accuracy)
    print(recall)
    print(precision)
    print(f1)


def main():
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    for gpu in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)
    # Running these will overwrite current data, ensure that you mean to do so before running it
    # read_and_save_audio()
    # speech_to_text(quick_run=True)
    # speech_to_text()

    audio_data = read_saved_audio()
    # print(audio_data)
    text_data = read_saved_text()

    train_audio_data, validation_audio_data = train_validation_split(audio_data)
    train_text_data, validation_text_data = train_validation_split(text_data)

    # train_and_run_lexicon_model_kfold(train_text_data)
    # train_and_run_similarity_model_kfold(train_text_data)
    # train_and_run_LSTM_model(train_audio_data)
    # train_and_run_LSTM_model(train_audio_data, model_file=MODEL_PATH, continue_training=True)
    # train_and_run_LSTM_model_kfold(train_audio_data)

    lstm = LSTM(MODEL_PATH)
    # validation_audio_data = split_sequence(validation_audio_data, SEQUENCE_LENGTH * 5)
    validation_audio_data = extend_sequence(validation_audio_data, CLIP_LENGTH * 3)
    x, y = convert_audio_data(validation_audio_data, CLIP_LENGTH * 3)
    # predictions = lstm.predict(x, batch_size=BATCH_SIZE)
    predictions = lstm.predict_long(x)
    print_statistics(y, predictions)


if __name__ == '__main__':
    main()
