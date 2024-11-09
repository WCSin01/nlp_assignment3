import pickle
from forward_backward_hmm import seed_matrices
from process_data import process_conllu
from forward_backward_hmm import forward_backward
import pickle
import numpy as np


CONTINUE = False

if __name__ == "__main__":
    if CONTINUE:
        f = open("checkpoints/word_to_one_hot.pkl", "rb")
        word_to_one_hot = pickle.load(f)
        dataset = process_conllu("ptb-train.conllu", word_to_one_hot)
        print("data parsed")
        pi = np.load("checkpoints/pi.npy").flatten()
        transition = np.load("checkpoints/transition.npy")
        emission_T = np.load("checkpoints/emission_T.npy")
        emission = emission_T.T
    else:
        dataset = process_conllu("ptb-train.conllu")
        f = open(f"checkpoints/word_to_one_hot.pkl", "wb")
        pickle.dump(dataset.ohe.word_to_one_hot, f)
        f.close()
        print("data parsed")
        pi, transition, emission = seed_matrices(len(dataset.upos), dataset.vocabulary_size, 1)

    output = forward_backward(dataset, pi, transition, emission, 1)
    f = open("checkpoints/output.pkl", "wb")
    pickle.dump(output, f)
