from dataclasses import dataclass
import numpy as np
import conllu


def process_conllu(file_path: str, word_to_one_hot: dict[str, int] = None):
    f = open(file_path, "r")
    data = f.read()
    data = conllu.parse(data)
    f.close()

    words = set()
    upos = set()
    xpos = set()
    # separated by <SEP>
    sentences = []
    # list of TokenList. Each TokenList is a sentence.
    # Token only prints form attribute. To obtain all, iterate over token like dictionary.
    for token_list in data:
        sentence = []
        for token in token_list:
            words.add(token["form"])
            upos.add(token["upos"])
            xpos.add(token["xpos"])
            sentence.append(token["form"])
        sentences.append(sentence)

    if not word_to_one_hot:
        word_to_one_hot = {word: i for i, word in enumerate(words)}
    ohe = OneHot(word_to_one_hot)
    return ConlluDataset(sentences, upos, xpos, ohe)


class OneHot:
    def __init__(self, word_to_one_hot: dict[str, int]):
        self.word_to_one_hot = word_to_one_hot
        self.one_hot_to_word: dict[int, str] = {v: k for k, v in word_to_one_hot.items()}
        self.one_hot_to_word[len(word_to_one_hot)] = "UNK"

    def encode(self, word: str):
        vector = np.zeros(len(self.word_to_one_hot)+1)
        if word in self.word_to_one_hot:
            vector[self.word_to_one_hot[word]] = 1
        else:
            vector[-1] = 1
        return vector

    def get_index(self, word: str):
        if word in self.word_to_one_hot:
            return self.word_to_one_hot[word]
        else:
            return len(self.word_to_one_hot)+1

    def decode(self, vector: np.ndarray) -> str:
        """

        :param vector: 1D
        :return:
        """
        index_where_1 = np.where(vector == 1)[0][0]
        return self.one_hot_to_word[index_where_1]

    def encode_row_log(self, word: str):
        matrix = np.full((1, len(self.word_to_one_hot)+1), -np.inf)
        if word in self.word_to_one_hot:
            matrix[0, self.word_to_one_hot[word]] = 0
        else:
            matrix[0, -1] = 0
        return matrix


@dataclass
class ConlluDataset:
    def __init__(self, sentences: list[list[str]], upos: set[str], xpos: set[str], ohe: OneHot):
        self.sentences = sentences
        self.upos = upos
        self.xpos = xpos
        self.ohe = ohe
        self.n_sentences = len(sentences)
        self.vocabulary_size = len(ohe.one_hot_to_word)


