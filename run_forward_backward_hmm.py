import pickle
from forward_backward_hmm import seed_matrices, HMM
from process_conllu import ConlluProcessor, ConlluDataset

if __name__ == "__main__":
    # dataset = ConlluProcessor.process_conllu_for_hmm("ptb-train.conllu")
    # f = open(f"checkpoints/dataset.pkl", "wb")
    # pickle.dump(dataset, f)
    # f.close()
    # print("data parsed")

    f = open("checkpoints/dataset.pkl", "rb")
    dataset: ConlluDataset = pickle.load(f)
    f.close()

    pi, transition, emission = seed_matrices(len(dataset.upos_set), dataset.vocabulary_size+1)
    hmm = HMM(dataset, pi, transition, emission)
    has_converged = hmm.forward_backward(max_iter=100)
    print(f"has convereged: {has_converged}")
