import numpy as np
from scipy.special import logsumexp
from forward_backward_hmm import forward_backward, seed_matrices
import pickle

# result = forward_backward(
#     np.array([[1,0,0,0],
#               [0,1,0,0],
#               [0,0,0,1],
#               [0,0,1,0],
#               [0,0,0,1]]),
#     np.array([1,0,0]),
#     np.array([[0.2, 0.6, 0.2],
#              [0.3, 0, 0.7],
#              [0, 0.5, 0.5]]),
#     np.array([[0.25, 0.25, 0.25, 0.25],
#              [0.3, 0.2, 0.3, 0.2],
#              [0.5, 0.4, 0.05, 0.05]])
# )
# print(np.sum(np.exp(result[0, :])))

# a = np.array([[[0.34955703, 0.58527719, 0.81751838, 0.78733667, 0.85623906]],
#  [[0.36551997, 0.98429999, 0.11092123, 0.28543393, 0.69633697]],
#  [[0.50409933, 0.3827965,  0.03202758, 0.71162717, 0.86186113]]])
# n = np.expand_dims(np.sum(a, axis=2), axis=2)
# an = a/n
# print(a)
# print(n)
# print(an)
# print("__")
# print(an[0])
# print(np.sum(an[0]))

# print(np.sum(a, axis=1))

# a = np.log(np.random.rand(3,5))
# b = np.log(np.random.rand(3,5))
# c = np.log(np.random.rand(3,5))
# ab = np.expand_dims(logsumexp(a+b, axis=1), axis=1)
# print(a)
# print(b)
# print(c)
# print(ab)
# print(c / ab)

# a = np.random.rand(3)
# b = np.random.rand(3,4)
# print(a)
# print(b)
# print(np.expand_dims(a, axis=1)+b)

# array = np.array([1, 2, 3, 4, 5])
# value = 3
# index = np.where(array == value)
# print(f"Index of value {value}: {index[0][0]}")

# seed_matrices(3, 5)

# print(np.log(0))
# print(np.exp(np.log(0)))

# a = np.array([
#     [[1,2,3]],
#     [[4,5,6]],
#     [[7,8,9]]
# ])
# print(a)
# b = a / np.expand_dims(np.sum(a, axis=0), axis=0)
# c = a / np.expand_dims(np.sum(a, axis=2), axis=2)
# print(b)
# print(np.sum(b, axis=0))
# print(c)
# print(np.sum(c, axis=0))

# print(np.sum([7.92778146e-04, 1.04461029e-01, 5.29526100e-05, 2.50079287e-01,
#   8.36249927e-03, 1.03482344e-02, 3.32473967e-03, 9.86535318e-02,
#   1.66573855e-02, 1.52852873e-01, 7.46425709e-02, 1.27271099e-01,
#   3.61979532e-02, 1.19196782e-02, 4.98068922e-03, 5.64772503e-02,
#   4.29254495e-02]))
from numeric import log_normalize
from process_data import process_conllu, OneHot

# POS = 17, V = 44390
# dataset = process_conllu("ptb-train.conllu")
# print(len(dataset.upos))

# f = open("checkpoints/word_to_one_hot.pkl", "rb")
# word_to_one_hot = pickle.load(f)
# # ohe = OneHot(word_to_one_hot)
# dataset = process_conllu("ptb-train.conllu", word_to_one_hot)
# ohe = dataset.ohe
# transition = np.load("checkpoints/transition.npy")
# # V x POS
# emission_T = np.load("checkpoints/emission_T.npy")
# assert np.allclose(emission_T.sum(axis=0), 1)
#
# for sentence in dataset.sentences:
#     for token in sentence:
#         u = emission_T[ohe.get_index(token)]
#         argmax_u = np.argmax(u)
#         print(argmax_u)
#     break

# print(np.argsort(transition[argmax_u])[::-1])
# w = emission_T[ohe.get_index("of")]
# print(w)

# a = np.random.rand(3, 4)
# d = np.log(a)
# d[0, 0] = -np.inf
# b = log_normalize(np.ma.masked_invalid(d), axis=0)
# print(b)
# c = a / np.expand_dims(np.sum(a, axis=0), axis=0)
# print(c)
