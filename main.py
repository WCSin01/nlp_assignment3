import pickle
from forward_backward_hmm import seed_matrices
from process_data import process_conllu
from forward_backward_hmm import forward_backward

if __name__ == "__main__":
    dataset = process_conllu("ptb-train.conllu")
    print("dataset parsed")
    pi, transition, emission = seed_matrices(len(dataset.upos), dataset.vocabulary_size, 1)
    pi, transition, emission = forward_backward(dataset, pi, transition, emission, 100_000)


