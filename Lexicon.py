import nltk
import numpy as np


class Lexicon:
    def __init__(self, train_x: np.ndarray = None, train_y: np.ndarray = None) -> None:
        self.author_vocabularies = dict()
        for (text, author) in zip(train_x, train_y):
            self.train_document(text, author)

    def train_document(self, document: str, author: str) -> None:
        # Due to speech to text translation, there is no punctuation or start/end of sentence markers
        if not self.author_vocabularies.get(author):
            self.author_vocabularies[author] = set()
        tokens = nltk.word_tokenize(document)
        self.author_vocabularies[author].update(tokens)

    def evaluate(self, test_x: np.ndarray, test_y: np.ndarray):
        count = 0
        predictions = []
        for (text, author) in zip(test_x, test_y):
            pred = self.classify(text)
            predictions.append(pred)
            if pred == author:
                count += 1
        return count / test_y.shape[0], np.array(predictions)

    def classify(self, test_instance: str) -> str:
        max_count = -1
        chosen_author = ""
        for (author, vocabulary) in self.author_vocabularies.items():
            count = 0
            for token in nltk.word_tokenize(test_instance):
                if token in vocabulary:
                    count += 1

            if count > max_count:
                chosen_author = author
                max_count = count

        return chosen_author
