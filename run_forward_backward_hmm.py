import numpy as np
from forward_backward_hmm import HMMParameters, seed_matrices, HMM
from functions import pickle_load
from process_conllu import ConlluDataset

# 35-45 min per iteration
if __name__ == "__main__":
    dataset: ConlluDataset = pickle_load("checkpoints/dataset.pkl")

    # exclude from training long sentences
    dataset.sequences = [seq for seq in dataset.sequences if len(seq) < 70]

    parameters: HMMParameters = seed_matrices(len(dataset.upos_set), dataset.vocabulary_size+1)

    parameters: HMMParameters = pickle_load("checkpoints/forward_backward_upos/epoch0.pkl")

    pi = parameters.pi
    transition = parameters.transition
    emission = parameters.emission

    # pi = np.array([1,0,0])
    # transition = np.array([[0,1,0],
    #                        [0,0,1],
    #                        [1,0,0]])
    # emission = np.array([[1,0,0],
    #                      [0,1,0],
    #                      [0,0,1]])
    # dataset.sequences = np.log(np.array([[[1,0,0],
    #                                       [0,1,0],
    #                                       [0,0,1]]]))

    hmm = HMM(dataset, pi, transition, emission)
    has_converged = hmm.forward_backward(max_iter=7)
    print(f"has convereged: {has_converged}")
