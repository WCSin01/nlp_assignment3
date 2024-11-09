import pickle
from forward_backward_hmm import seed_matrices
from process_data import process_conllu
from forward_backward_hmm import forward_backward
import pickle


if __name__ == "__main__":
    dataset = process_conllu("ptb-train.conllu")
    print("dataset parsed")
    f = open(f"checkpoints/word_to_one_hot.pkl", "wb")
    pickle.dump(dataset.ohe.word_to_one_hot, f)
    f.close()
    pi, transition, emission = seed_matrices(len(dataset.upos), dataset.vocabulary_size, 1)
    output = forward_backward(dataset, pi, transition, emission, 100)
