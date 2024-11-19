import numpy as np
from forward_backward_hmm import HMMParameters, seed_matrices, HMM
from functions import pickle_load
from process_conllu import ConlluDataset, OneHot, Dataset

# 35-45 min per iteration
if __name__ == "__main__":
    # dataset: ConlluDataset = pickle_load("checkpoints/dataset.pkl")

    # exclude from training long sentences
    # dataset.sequences = [seq for seq in dataset.sequences if len(seq) < 70]

    # parameters: HMMParameters = seed_matrices(len(dataset.upos_set), dataset.vocabulary_size+1)

    # parameters: HMMParameters = pickle_load("checkpoints/forward_backward_upos/epoch0.pkl")

    pi = np.array([1,0,0])
    transition = np.array([[0,1,0],
                           [0,0,1],
                           [1,0,0]])
    emission = np.array([[1,0,0],
                         [0,1,0],
                         [0,0,1]])
    dataset = Dataset([[0,1,2]], OneHot([0,1,2]))

    hmm = HMM(seed_matrices(3,4), dataset, "checkpoints/forward_backward")
    has_converged = hmm.forward_backward(max_iter=30)
    print("PI")
    print(np.exp(hmm.log_pi))
    print("TRANSITION")
    print(np.exp(hmm.log_transition))
    print("EMISSION")
    print(np.exp(hmm.log_emission_T.T))
    print(f"has convereged: {has_converged}")
