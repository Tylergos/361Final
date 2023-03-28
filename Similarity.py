import math

import nltk
import numpy as np


def dotprod(vector1, vector2):
    # Dot product from https://stackoverflow.com/a/43540715
    return sum([x * y for x, y in zip(vector1, vector2)])


class Similarity:
    def __init__(self, train_x: np.ndarray = None, train_y: np.ndarray = None) -> None:
        print("Print training")
        self.embeddings: dict[str, dict[str, float]] = dict()
        self.unique_words: set = set()
        for (text, author) in zip(train_x, train_y):
            self.train_document(text, author)

    def train_document(self, document: str, cur_author: str) -> None:
        doc_set = set()
        author_dict = self.embeddings.get(cur_author, {})
        if author_dict == {}:
            self.embeddings[cur_author] = {}

        for token in nltk.word_tokenize(document):
            self.unique_words.add(token)

            if token not in doc_set:
                doc_set.add(token)
                self.embeddings[cur_author][token] = author_dict.get(token, 0) + 1

    def evaluate(self, test_x: np.ndarray, test_y: np.ndarray):
        predictions = []
        count = 0
        num = 1
        total = test_y.shape[0]
        for (text, author) in zip(test_x, test_y):
            pred = self.classify(text)
            predictions.append(pred)
            if pred == author:
                count += 1
            print("Done classifying " + str(num) + " out of " + str(total))
            num += 1
        return count / test_y.shape[0], np.array(predictions)

    def classify(self, test_instance: str) -> str:
        max_sim: float = -1.0
        chosen_author: str = ""

        test_dict: dict[str, float] = {}
        test_set = set()
        test_tokens = nltk.word_tokenize(test_instance)

        for token in test_tokens:
            if token not in test_set:
                if token in self.unique_words:
                    test_set.add(token)
                    test_dict[token] = 1

        author: str
        vocab: dict
        for (author, vocab) in self.embeddings.items():
            dot: float = 0
            for token in test_set:
                if token in vocab:
                    dot += (vocab[token] * test_dict[token])

            length_sample = math.sqrt(dotprod(vocab.values(), vocab.values()))
            length_test = math.sqrt(dotprod(test_dict.values(), test_dict.values()))

            sample_sim = dot / (length_sample * length_test)

            if sample_sim > max_sim:
                max_sim = sample_sim
                chosen_author = author
        return chosen_author
