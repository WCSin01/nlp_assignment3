import numpy as np
from scipy.special import logsumexp
from forward_backward_hmm import forward_backward, seed_matrices

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


