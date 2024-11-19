import numpy as np
from forward_backward_hmm import HMMParameters, HMM
from process_conllu import OneHot, Dataset

if __name__ == "__main__":
    pi = np.array([1,0,0])
    transition = np.array([[0,1,0],
                           [0,0,1],
                           [1,0,0]])
    emission = np.array([[1,0,0],
                         [0,1,0],
                         [0,0,1]])
    dataset = Dataset([[0,1,2]], OneHot([0,1,2]))

    hmm = HMM(HMMParameters(pi, transition, emission), dataset, "checkpoints/forward_backward_test")
    has_converged = hmm.forward_backward(max_iter=30)
    print("PI")
    print(np.exp(hmm.log_pi))
    print("TRANSITION")
    print(np.exp(hmm.log_transition))
    print("EMISSION")
    print(np.exp(hmm.log_emission_T.T))
    print(f"has convereged: {has_converged}")
