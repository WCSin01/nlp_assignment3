from dataclasses import dataclass
import numpy as np
import conllu
from conllu import TokenList


def process_conllu(file_path: str):
    f = open(file_path, "r")
    data = f.read()
    data = conllu.parse(data)
    f.close()

    words = set()
    upos = set()
    xpos = set()
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

    def decode(self, vector: np.ndarray) -> str:
        """

        :param vector: 1D
        :return:
        """
        index_where_1 = np.where(vector == 1)[0][0]
        return self.one_hot_to_word[index_where_1]


@dataclass
class ConlluDataset:
    def __init__(self, data: list[TokenList], upos: set[str], xpos: set[str], ohe: OneHot):
        self.data = data
        self.upos = upos
        self.xpos = xpos
        self.ohe = ohe
        self.dataset_size = len(data)
        self.vocabulary_size = len(ohe.one_hot_to_word)


