import nltk


class Lexicon:
    def __init__(self, author_vocabularies: dict[list[str]] = None) -> None:
        self.author_vocabularies = dict()
        for (author, texts) in author_vocabularies.items():
            self.train_author_lexicon(author, texts)

    def train_author_lexicon(self, author: str, texts: list[str]) -> None:
        # Due to speech to text translation, there is no punctuation or start/end of sentence markers
        lexicon = set()
        for text in texts:
            tokens = nltk.word_tokenize(text)
            lexicon.update(tokens)
        self.author_vocabularies[author] = lexicon

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
