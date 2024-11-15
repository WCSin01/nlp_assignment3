from forward_backward_hmm import HMMParameters, seed_matrices, HMM
from functions import pickle_load
from process_conllu import ConlluDataset

# 35-45 min per iteration
if __name__ == "__main__":
    dataset: ConlluDataset = pickle_load("checkpoints/dataset.pkl")

    # pi, transition, emission = seed_matrices(len(dataset.upos_set), dataset.vocabulary_size+1)

    parameters: HMMParameters = pickle_load("checkpoints/forward_backward/epoch4.pkl")

    pi = parameters.pi.flatten()
    transition = parameters.transition
    emission = parameters.emission

    hmm = HMM(dataset, pi, transition, emission)
    has_converged = hmm.forward_backward(max_iter=5)
    print(f"has convereged: {has_converged}")
