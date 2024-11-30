from typing import TypeVar, Iterable, Generic
import re
from dataclasses import dataclass
import numpy as np
import conllu

T = TypeVar("T")


class ConlluProcessor:
    @staticmethod
    def process_conllu_for_inspection(file_path: str):
        f = open(file_path, "r")
        data = f.read()
        data = conllu.parse(data)
        f.close()

        word_counts = dict()
        sentence_lengths = []
        # list of TokenList. Each TokenList is a sentence.
        # Token only prints form attribute. To obtain all, iterate over token like dictionary.
        for token_list in data:
            sentence_lengths.append(len(token_list))
            for token in token_list:
                # ordered set
                word = ConlluProcessor.preprocess_word(token["form"])
                if word in word_counts:
                    word_counts[word] += 1
                else:
                    word_counts[word] = 1
        return word_counts, sentence_lengths

    @staticmethod
    def process_conllu_for_bert(file_path: str):
        f = open(file_path, "r")
        data = f.read()
        data = conllu.parse(data)
        f.close()

        n_tokens = 0
        sentences: list[list[str]] = []
        # list of TokenList. Each TokenList is a sentence.
        # Token only prints form attribute. To obtain all, iterate over token like dictionary.
        for token_list in data:
            sentence = []
            # filter sentences
            if len(token_list) < 3 and token_list[-1]["form"] not in [".", "?", "!"]:
                continue
            for token in token_list:
                n_tokens += 1
                sentence.append(token["form"])
            sentences.append(sentence)

        return sentences

    @staticmethod
    def process_conllu_for_hmm(file_path: str, filter_count: int):
        """

        :param file_path:
        :param filter_count: keep word in vocabulary if count > filter_count
        :return:
        """
        f = open(file_path, "r")
        data = f.read()
        data = conllu.parse(data)
        f.close()

        word_counts = dict()
        upos_set = set()
        xpos_set = set()
        n_tokens = 0
        sentences = []
        upos = []
        xpos = []
        # list of TokenList. Each TokenList is a sentence.
        # Token only prints form attribute. To obtain all, iterate over token like dictionary.
        for token_list in data:
            sentence = []
            sentence_upos = []
            sentence_xpos = []
            # filter sentences
            if len(token_list) < 3 and token_list[-1]["form"] not in [".", "?", "!"]:
                continue
            for token in token_list:
                n_tokens += 1
                # ordered set
                word = ConlluProcessor.preprocess_word(token["form"])
                if word in word_counts:
                    word_counts[word] += 1
                else:
                    word_counts[word] = 1
                upos_set.add(token["upos"])
                sentence_upos.append(token["upos"])
                xpos_set.add(token["xpos"])
                sentence_xpos.append(token["xpos"])
                sentence.append(word)
            sentences.append(sentence)
            upos.append(sentence_upos)
            xpos.append(sentence_xpos)

        # remove low counts
        filtered_words = [word for word in word_counts if word_counts[word] > filter_count]
        ohe = OneHot(filtered_words)
        return ConlluDataset(sentences, ohe, upos, xpos, upos_set, xpos_set, n_tokens)

    @staticmethod
    def preprocess_word(word: str):
        # TODO: discard extremely short sentences, handle -lcb- etc.
        if any(char.isdigit() for char in word):
            word = word.replace(",", "").replace(".", "")
            word = re.sub(r"\d+", "[NUM]", word)
            return word
        elif "$" in word:
            return "$"
        elif word in ["*", "**", "#", "="]:
            return "[SYM]"
        elif word == ".what":
            return "what"
        else:
            return word.lower()


class OneHot(Generic[T]):
    def __init__(self, values: Iterable[T]):
        self.to_index: dict[T, int] = {value: i for i, value in enumerate(sorted(values))}
        self.from_index: dict[int, T] = {v: k for k, v in self.to_index.items()}
        self.from_index[len(values)] = "UNK"

    def encode(self, value: T):
        vector = np.zeros(len(self.to_index) + 1)
        if value in self.to_index:
            vector[self.to_index[value]] = 1
        else:
            vector[-1] = 1
        return vector

    def get_index(self, value: T):
        if value in self.to_index:
            return self.to_index[value]
        else:
            return len(self.to_index)

    def decode(self, vector: np.ndarray):
        """

        :param vector: 1D
        :return:
        """
        index_where_1 = np.where(vector == 1)[0][0]
        return self.from_index[index_where_1]
    
    def encode_log(self, value: T):
        vector = np.full(len(self.to_index) + 1, -np.inf)
        if value in self.to_index:
            vector[self.to_index[value]] = 0
        else:
            vector[-1] = 0
        return vector
    
    def encode_row_log(self, value: T):
        return value
    
    # def encode_row_log(self, value: T):
    #     matrix = np.full((1, len(self.to_index) + 1), -np.inf)
    #     if value in self.to_index:
    #         matrix[0, self.to_index[value]] = 0
    #     else:
    #         matrix[0, -1] = 0
    #     return matrix


@dataclass
class Dataset(Generic[T]):
    def __init__(self, sequences: list[list[T]], ohe: OneHot):
        self.sequences = sequences
        self.ohe = ohe


@dataclass
class ConlluDataset(Dataset):
    def __init__(self, sequences: list[list[str]], ohe: OneHot, upos: list[list[str]], xpos: list[list[str]],
                 upos_set: set[str], xpos_set: set[str], n_tokens: int):
        super().__init__(sequences, ohe)
        self.upos = upos
        self.xpos = xpos
        self.upos_set = upos_set
        self.xpos_set = xpos_set
        self.n_sentences = len(sequences)
        self.n_tokens = n_tokens
        self.vocabulary_size = len(ohe.from_index)
