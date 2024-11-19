import numpy as np
from forward_backward_hmm import HMMParameters, seed_matrices, HMM
from functions import pickle_load
from process_conllu import ConlluDataset, OneHot, Dataset

# 35-45 min per iteration
if __name__ == "__main__":
    dataset: ConlluDataset = pickle_load("checkpoints/dataset.pkl")

    # exclude from training long sentences
    dataset.sequences = [seq for seq in dataset.sequences if len(seq) < 70]

    # parameters: HMMParameters = seed_matrices(len(dataset.upos_set), dataset.vocabulary_size)

    parameters: HMMParameters = pickle_load("checkpoints/forward_backward_upos/epoch6.pkl")

    hmm = HMM(parameters, dataset, "checkpoints/forward_backward_upos_running")
    has_converged = hmm.forward_backward(max_iter=1)
    print(f"has convereged: {has_converged}")
