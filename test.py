import numpy as np
from process_conllu import OneHot, ConlluDataset
from forward_backward_hmm import HMM

# f = open("checkpoints/dataset.pkl", "rb")
# dataset: ConlluDataset = pickle.load(f)
# print(f"n_sentences: {dataset.n_sentences}")
# print(f"n_tokens: {dataset.n_tokens}")
# print(f"vocabulary size: {dataset.vocabulary_size}")
# f.close()
#
# pi, transition, emission = seed_matrices(len(dataset.upos_set), dataset.vocabulary_size, seed=1)

rolls = [2, 0, 3, 1, 0, 5, 4, 3, 5, 4]
rolls_one_hot = np.full((len(rolls), 1, 6), -324)
for idx, val in enumerate(rolls):
    rolls_one_hot[idx, 0, val] = 0
dataset = ConlluDataset([rolls_one_hot], OneHot({}), None, None, None, None, None)
hmm = HMM(dataset,
          np.array([1.0, 0]),
          np.array([[0.95, 0.05],
                    [0.1, 0.9]]),
          np.array([[1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6],
                    [1 / 10, 1 / 10, 1 / 10, 1 / 10, 1 / 10, 1 / 2]])
          )
print(hmm.forward_backward(max_iter=1))

