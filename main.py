import math
import random
import sys

import librosa
import matplotlib.pyplot as plt
import os

import numpy
import numpy as np
import pickle
import speech_recognition as sr
import nltk
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
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
    # Just reads in the audio data while also saving the data in a dictionary
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
    # Loads in the saved dictionary of speaker audio data
    file = open(file_name, "rb")
    audio_data = pickle.load(file)
    file.close()
    return audio_data


def save_train_set(train_set=list[list[int]], file_name: str = TRAIN_SET_PATH) -> None:
    # Saves the split of speaker audio data
    pickle_file = open(file_name, "wb")
    pickle.dump(train_set, pickle_file)
    pickle_file.close()


def read_train_set(file_name: str = TRAIN_SET_PATH) -> list[list[int]]:
    # Loads the split of speaker audio data
    file = open(file_name, "rb")
    train_set = pickle.load(file)
    file.close()
    return train_set


def train_validation_split(data: dict[list[any]]) -> dict[list[any]]:
    # Splits the dataset into a train split and validation split. 80% train, 20% validation
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
    # Converts the audio file into text format and saves the text files in a dictionary
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
    # Loads the saved text dataset
    file = open(file_name, "rb")
    text_data = pickle.load(file)
    file.close()
    return text_data


def convert_audio_data(audio_data: dict[list[str]], clip_length: int) -> tuple[np.ndarray, np.ndarray]:
    # Convert the audio data to be in a format that keras model can read easier
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


def split_sequence(audio_data: dict[list[float]], sequence_length: int) -> dict[list[float]]:
    # Splits the audio data into smaller samples
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


def convert_text_data(text_data: dict[list[str]]) -> tuple[np.ndarray, np.ndarray]:
    # Converts the text data into a format to be easier to use
    x = []
    y = []
    for (author, data) in text_data.items():
        for document in data:
            # don't need the sample rate, thus we only take index 0
            x.append(document)
            y.append(int(author[-2::]))

    return np.array(x), np.array(y)


def train_and_run_LSTM_model(train_data: dict[list[str]]) -> None:
    # Makes, trains, and evaluates the LSTM model
    data = split_sequence(train_data, SEQUENCE_LENGTH)
    x, y = convert_audio_data(data, SEQUENCE_LENGTH)

    train_x, test_x, train_y, test_y = train_test_split(x, y, train_size=0.8, random_state=RANDOM_STATE)

    lstm_model = LSTM()
    lstm_model.train_model(train_x, train_y, test_x, test_y, steps_per_epoch=len(train_y) // BATCH_SIZE,
                           batch_size=BATCH_SIZE, output_file=MODEL_PATH, epochs=EPOCHS)
    lstm_model.evaluate(test_x, test_y)


def train_and_run_lexicon_model(train_data: dict[list[str]], test_data: dict[list[str]], full: bool) -> np.ndarray:
    # Makes, trains, and evaluates  the Lexicon model
    train_x, train_y = convert_text_data(train_data)
    test_x, test_y = convert_text_data(test_data)

    # Sometimes when combining/splitting text data, certain authors would have blank text
    # Due to the implementation before, this gets around that but is only needed in certain situations
    # i.e. when testing model on shorter clips
    if full:
        test_y = test_y[np.where(test_x != "")]
        test_x = test_x[np.where(test_x != "")]

    lexicon_model = Lexicon(train_x, train_y)
    accuracy, predictions = lexicon_model.evaluate(test_x, test_y)
    print("Accuracy")
    print(accuracy)
    print("Recall")
    print(recall_score(y_true=test_y, y_pred=predictions, average="macro"))
    print("Precision")
    print(precision_score(y_true=test_y, y_pred=predictions, average="macro"))
    print("F1-Score")
    print(f1_score(y_true=test_y, y_pred=predictions, average="macro"))
    return predictions


def train_and_run_similarity_model(train_data: dict[list[str]], test_data: dict[list[str]], full: bool) -> np.ndarray:
    # Makes, trains, and evaluates  the Similarity model
    train_x, train_y = convert_text_data(train_data)
    test_x, test_y = convert_text_data(test_data)

    # Sometimes when combining/splitting text data, certain authors would have blank text
    # Due to the implementation before, this gets around that but is only needed in certain situations
    # i.e. when testing model on shorter clips
    if full:
        test_y = test_y[np.where(test_x != "")]
        test_x = test_x[np.where(test_x != "")]

    similarity_model = Similarity(train_x, train_y)
    accuracy, predictions = similarity_model.evaluate(test_x, test_y)
    print("Accuracy")
    print(accuracy)
    print("Recall")
    print(recall_score(y_true=test_y, y_pred=predictions, average="macro"))
    print("Precision")
    print(precision_score(y_true=test_y, y_pred=predictions, average="macro"))
    print("F1-Score")
    print(f1_score(y_true=test_y, y_pred=predictions, average="macro"))
    return predictions


def sample_distribution(text_data: dict[list[str]]) -> None:
    # Just plot the distribution of samples
    x, y = convert_text_data(text_data)
    counts = dict()

    for sample in y:
        counts[sample] = counts.get(sample, 0) + 1

    plt.bar(list(counts.keys()), counts.values(), color='g')
    plt.xlabel("Author")
    plt.ylabel("Number of Samples")
    plt.title("Text Sample Distribution Per Author")
    plt.show()


def text_extend_and_split(text_data, samples_per_clip):
    # Combine all the text for an author together and then split into SAMPLES_PER_CLIP number of different samples to
    # Represent the text for different length validation data
    num_clips = 0
    for (author, samples) in text_data.items():
        num_clips = len(samples)
        combined_text = [""]
        for sample in samples:
            combined_text[0] = combined_text[0] + " " + sample
        if combined_text == [""]:
            continue
        text_data[author] = combined_text
        text_split(text_data, samples_per_clip * num_clips, author, combined_text)


def text_split(text_data, num_splits, author, author_text):
    # Splits the data into a certain number of splits. If the last split is not full, ignore it
    text_data[author] = []
    author_text = nltk.word_tokenize(author_text[0])
    num_total_words = len(author_text)
    num_split_words = math.ceil(num_total_words / num_splits)

    cur_split_num = 1
    cur_num_words_split = 0
    cur_split = ""
    for word in author_text:
        cur_split += " " + word
        cur_num_words_split += 1
        if cur_num_words_split == num_split_words:
            cur_split_num += 1
            text_data[author] = text_data.get(author, []) + [cur_split]
            cur_split = ""
            cur_num_words_split = 0
            if cur_split_num == num_splits:
                break


def main():
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    for gpu in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)

    # Running these will overwrite current data, ensure that you mean to do so before running it
    # read_and_save_audio()
    # speech_to_text(quick_run=True)
    # speech_to_text()

    audio_data = read_saved_audio()
    text_data = read_saved_text()

    sample_distribution(text_data)
    train_audio_data, validation_audio_data = train_validation_split(audio_data)
    train_text_data, validation_text_data = train_validation_split(text_data)

    # Second parameter 2 to get 30s samples or 10 to get 6s samples
    text_extend_and_split(validation_text_data, 2)
    print("Lexicon")
    train_and_run_lexicon_model(train_text_data, validation_text_data, full=True)


    print("Similarity")
    train_and_run_similarity_model(train_text_data, validation_text_data, full=True)
    train_and_run_LSTM_model(train_audio_data)

    sample_distribution(text_data)

    lstm = LSTM(MODEL_PATH)
    data = split_sequence(validation_audio_data, SEQUENCE_LENGTH)
    x, y = convert_audio_data(data, SEQUENCE_LENGTH)
    lstm.evaluate(x, y, BATCH_SIZE)


if __name__ == '__main__':
    main()
