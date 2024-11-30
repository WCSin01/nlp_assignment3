import numpy as np
from forward_backward_hmm import HMMParameters, seed_matrices, HMM
from functions import pickle_load
from process_conllu import ConlluDataset, OneHot, Dataset


if __name__ == "__main__":
    dataset: ConlluDataset = pickle_load("checkpoints/dataset.pkl")

    # exclude from training long sentences
    dataset.sequences = [seq for seq in dataset.sequences if len(seq) < 70]

    parameters: HMMParameters = seed_matrices(len(dataset.xpos_set), dataset.vocabulary_size)

    # parameters: HMMParameters = pickle_load("checkpoints/forward_backward_xpos/epoch170.pkl")

    hmm = HMM(parameters, dataset, "checkpoints/forward_backward_xpos_running")
    has_converged = hmm.forward_backward(max_iter=200)
    print(f"has converged: {has_converged}")
