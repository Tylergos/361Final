import sys

import librosa
import os

import numpy
import numpy as np
import pickle
import speech_recognition as sr
import nltk
from sklearn.model_selection import StratifiedKFold, train_test_split
import tensorflow as tf

from Lexicon import Lexicon
from LSTM import LSTM

from keras.utils import pad_sequences

DATASET_PATH = "./SpeakersDataset/50_speakers_audio_data"
PICKLE_AUDIO_FILE_PATH = "./AudioData.txt"
PICKLE_TEXT_FILE_PATH = "./TextData.txt"
MODEL_PATH = "model.h5"
SAMPLE_RATE = 11025 // 8
CLIP_LENGTH = 661504 // 8
SEQUENCES_PER_CLIP = 10
SEQUENCE_LENGTH = CLIP_LENGTH // SEQUENCES_PER_CLIP
BATCH_SIZE = 8
EPOCHS = 120


def read_and_save_audio(file_name: str = PICKLE_AUDIO_FILE_PATH) -> None:
    speaker_files = dict()
    for speaker in os.listdir(DATASET_PATH):
        audio_files = list()
        for audio_file in os.listdir(DATASET_PATH + "/" + speaker):
            print(audio_file)
            # open and resample the files from ~22Khz to ~11KHz.
            try:
                audio_files.append(librosa.load(DATASET_PATH + "/" + speaker + "/" + audio_file, sr=SAMPLE_RATE))
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
            except: # file is corrupted or has issues
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


def convert_audio_data(audio_data: dict[list[str]], clip_length: int) -> tuple[np.ndarray, np.ndarray]:
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


def convert_text_data(text_data: dict[list[str]]) -> tuple[np.ndarray, np.ndarray]:
    x = []
    y = []
    for (author, data) in text_data.items():
        for document in data:
            # don't need the sample rate, thus we only take index 0
            x.append(document)
            y.append(int(author[-2::]))

    return np.array(x), np.array(y)


def train_and_run_LSTM_model(audio_data: dict[list[float]]) -> None:
    data = split_sequence(audio_data, SEQUENCE_LENGTH)
    x, y = convert_audio_data(data, SEQUENCE_LENGTH)

    train_x, test_x, train_y, test_y = train_test_split(x, y, train_size=0.8)

    lstm_model = LSTM()
    lstm_model.train_model(train_x, train_y, test_x, test_y, steps_per_epoch=len(train_y) // BATCH_SIZE, batch_size=BATCH_SIZE, output_file=MODEL_PATH, epochs=EPOCHS)
    lstm_model.evaluate(test_x, test_y)


def train_and_run_LSTM_model_kfold(audio_data: dict[list[float]]) -> None:
    x, y = convert_audio_data(audio_data)

    kfold = StratifiedKFold(5)

    accuracies = []
    for fold, (train_index, test_index) in enumerate(kfold.split(x, y)):
        train_x = x[train_index]
        train_y = y[train_index]
        test_x = x[test_index]
        test_y = y[test_index]

        lstm_model = LSTM()
        lstm_model.train_model(train_x, train_y, test_x, test_y, steps_per_epoch=len(train_y) // BATCH_SIZE, batch_size=BATCH_SIZE)
        accuracies.append(lstm_model.evaluate(test_x, test_y))
    print(accuracies)


def train_and_run_lexicon_model_kfold(text_data: dict[list[str]]) -> None:
    x, y = convert_text_data(text_data)

    kfold = StratifiedKFold(5)

    accuracies = []
    for fold, (train_index, test_index) in enumerate(kfold.split(x, y)):
        train_x = x[train_index]
        train_y = y[train_index]
        test_x = x[test_index]
        test_y = y[test_index]

        lexicon_model = Lexicon(train_x, train_y)
        accuracy, _ = lexicon_model.evaluate(test_x, test_y)
        accuracies.append(accuracy)

    print(np.sum(accuracies) / 5)


def split_sequence(audio_data: dict[list[float]], sequence_length: int) -> dict[list[float]]:
    sequence_data = dict()
    for (author, data) in audio_data.items():
        sequences = list()
        for clip in data:
            i = 0
            while i + sequence_length < len(clip[0]):
                sequences.append(clip[0][i:i + sequence_length])
                i += sequence_length
        sequence_data[author] = sequences

    return sequence_data


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
    # text_data = read_saved_text()

    # train_and_run_lexicon_model_kfold(text_data)
    train_and_run_LSTM_model(audio_data)
    # train_and_run_LSTM_model_kfold(audio_data)

    # lstm = LSTM(MODEL_PATH)
    # x, y = convert_audio_data(audio_data)
    # lstm.evaluate(x, y)


if __name__ == '__main__':
    main()
