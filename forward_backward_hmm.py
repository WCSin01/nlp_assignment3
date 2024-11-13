from dataclasses import dataclass
from typing import Generic, TypeVar

import numpy as np
from sklearn.metrics import mean_squared_error
from scipy.special import logsumexp
from numeric import log_normalize, log_mat_mul
from process_conllu import Dataset

TypeT = TypeVar("T")
min_float = 1e-323


@dataclass
class ForwardBackwardOutput:
    def __init__(
            self,
            pi: np.ndarray,
            transition: np.ndarray,
            emission: np.ndarray,
            has_converged: bool):
        self.pi = pi
        self.transition = transition
        self.emission = emission
        self.has_converged = has_converged


class HMM(Generic[TypeT]):
    def __init__(self, dataset: Dataset[TypeT], pi: np.ndarray, transition: np.ndarray, emission: np.ndarray):
        """

        :param dataset:
        :param pi: initial probability distribution. |POS|
        :param transition: transition matrix. axis=0: current state, axis=1: next state. |POS|^2
        :param emission: emission matrix. axis=0: hidden state, axis=1: observed state. |POS| x |vocabulary|
        """
        POS, V = emission.shape
        assert POS == pi.shape[0]
        assert (POS, POS) == transition.shape
        assert np.allclose(pi.sum(), 1)
        assert np.allclose(transition.sum(axis=1), 1)
        assert np.allclose(emission.sum(axis=1), 1)

        self.dataset = dataset
        # np.log(0) = -np.inf
        with np.errstate(divide='ignore', invalid='raise'):
            # increase dimension for mat_mul
            self.log_pi = np.expand_dims(np.log(pi), axis=0)
            self.log_transition = np.log(transition)
            self.log_transition_T = self.log_transition.T
            self.log_emission_T = np.log(emission).T

    def update(self, pi, transition, emission_T):
        self.log_pi = np.log(pi)
        self.log_transition = np.log(transition)
        self.log_transition_T = self.log_transition.T
        self.log_emission_T = emission_T

    def forward_backward(
            self,
            max_iter: int) -> ForwardBackwardOutput:
        """

        :param max_iter: maximum iteration to stop if no convergence.
        :return: pi, transition, emission, if_converged
        """
        with np.errstate(divide='ignore', invalid='raise'):
            # one outer loop is one iteration through the whole dataset
            for i in range(max_iter):
                for j, sequence in enumerate(self.dataset.sequences):
                    # encode word only when required to reduce memory usage
                    p_j_at_t, xi = self.forward_backward_expect(sequence)

                    np.save(f"checkpoints/p_j_at_t/sequence{j}", p_j_at_t)
                    np.save(f"checkpoints/xi/sequence{j}", xi)

                    if j % 500 == 0:
                        print(f"epoch: {i + 1}, sequence #: {j + 1}/{len(self.dataset.sequences)}")

                # TODO
                new_pi, new_transition, new_emission_T = self.forward_backward_max(
                    sequence, p_j_at_t, xi, self.log_emission_T.shape[0])

                assert np.allclose(np.sum(new_transition, axis=1), 1)
                assert np.allclose(np.sum(new_emission_T, axis=0), 1)

                transition = np.exp(self.log_transition)
                print(f"MSE: {mean_squared_error(new_transition.flatten(), transition.flatten())}")

                if (np.allclose(transition, new_transition) and
                        np.allclose(np.exp(self.log_emission_T), new_emission_T)):
                    self.update(new_pi, new_transition, new_emission_T)
                    return ForwardBackwardOutput(new_pi, new_transition, new_emission_T.T, True)
                else:
                    self.update(new_pi, new_transition, new_emission_T)

            return ForwardBackwardOutput(new_pi, new_transition, new_emission_T.T, False)

    def log_forward(self, sequence: list[TypeT]) -> np.ndarray:
        """

        :param sequence:
        :return: log a_t(j) = log P(o_1 o_2 ... o_t, q_t=j|lambda). Tx1x|POS|
        """
        with np.errstate(divide='ignore', invalid='raise'):
            T = len(sequence)
            _, POS = self.log_pi.shape
            ohe = self.dataset.ohe

            # initialize
            # length x POS
            log_p_forward = np.full((T, 1, POS), 1e-500)

            # 1 x V @ V x POS = 1 x POS
            # 1 x POS * 1 x POS
            log_p_forward[0] = self.log_pi + log_mat_mul(ohe.encode_row_log(sequence[0]), self.log_emission_T)

            # recursion
            for t in range(1, T):
                # 1 x POS @ POS x POS = 1 x POS
                log_p_forward[t] = log_mat_mul(log_p_forward[t - 1], self.log_transition) + \
                                   log_mat_mul(ohe.encode_row_log(sequence[t]), self.log_emission_T)
            return log_p_forward

    def log_backward(self, sequence: list[TypeT]) -> np.ndarray:
        """

        :param sequence:
        :return: log b_t(i) = log P(o_t+1 o_t+2 ... o_T|q_t = i, lambda). Tx1x|POS|
        """
        T = len(sequence)
        _, POS = self.log_pi.shape
        ohe = self.dataset.ohe

        # initialize
        log_p_backward = np.full((T, 1, POS), -np.inf)
        log_p_backward[-1] = np.zeros((1, POS))  # this is not a probability distribution

        # recursion
        # from T-2 to 0 inclusive
        for t in range(T - 2, -1, -1):
            # 1 x V @ V x POS = 1 x POS
            # 1 x POS @ POS x POS = 1 x POS
            log_p_backward[t] = log_mat_mul(
                log_mat_mul(
                    ohe.encode_row_log(sequence[t + 1]),
                    self.log_emission_T
                ),
                self.log_transition_T
            ) + log_p_backward[t + 1, :]
        return log_p_backward

    def expected_state_occupancy_count(
            self,
            log_p_forward: np.ndarray,
            log_p_backward: np.ndarray) -> np.ndarray:
        """

        :param log_p_forward:
        :param log_p_backward:
        :return: gamma, P(q_t=j|O, lambda). T x 1 x |POS|
        """
        p_j_at_t = log_normalize(log_p_forward + log_p_backward, axis=2)
        return p_j_at_t

    def expected_state_transition_count(
            self,
            sequence: list[TypeT],
            log_p_forward: np.ndarray,
            log_p_backward: np.ndarray) -> np.ndarray:
        """

        :param sequence:
        :param log_p_forward:
        :param log_p_backward:
        :return: xi_t(ij).
            P(q_t=i, q_{t+1}=j|O, lambda).
            the probability at state i at time t and state j at time t+1.
            (T-1) x |POS| x |POS|
        """
        T, _, POS = log_p_forward.shape
        log_xi = np.zeros((T - 1, POS, POS))
        ohe = self.dataset.ohe

        # for t in range(T - 1):
        #     # axis 0: from state, axis 1: to state
        #     # POS x 1 * POS x POS * (1 x V @ V x POS) * 1 x POS
        #     log_xi[t] = log_p_forward[t].T + self.log_transition +\
        #                 log_mat_mul(ohe.encode_row_log(sequence[t + 1]), self.log_emission_T) +\
        #                 log_p_backward[t + 1]

        # equivalent to:
        for t in range(T - 1):
            b_j = log_mat_mul(ohe.encode_row_log(sequence[t+1]), self.log_emission_T)[0]
            for i in range(POS):
                for j in range(POS):
                    log_xi[t, i, j] = log_p_forward[t, 0, i] + self.log_transition[i, j] +\
                                      b_j[j] + log_p_backward[t+1, 0, j]

        # normalize s.t. every time unit sums to 1
        xi = log_normalize(log_xi, axis=(1,2))
        return xi

    def forward_backward_expect(self, sequence: list[TypeT]) -> tuple[np.ndarray, np.ndarray]:
        """

        :param sequence:
        :return: p_j_at_t, xi
        """
        log_p_forward = self.log_forward(sequence)
        log_p_backward = self.log_backward(sequence)
        # p_o_lambda: P(O|lambda)
        # p_o_lambda = np.sum(log_forward_mat[-1])
        # (T-1) x 1 x 1
        p_j_at_t = self.expected_state_occupancy_count(log_p_forward, log_p_backward)
        xi = self.expected_state_transition_count(
            sequence, log_p_forward, log_p_backward)
        return p_j_at_t, xi

    def forward_backward_max(
            self,
            sequence: list[TypeT],
            p_j_at_t: np.ndarray,
            xi: np.ndarray,
            vocabulary_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """

        :param sequence:
        :param p_j_at_t:
        :param xi:
        :param vocabulary_size:
        :return: pi, transition, emission_T
        """
        ohe = self.dataset.ohe
        # log_transition: np.ndarray = logsumexp(np.log(xi), axis=0)
        # transition = log_normalize(log_transition, axis=1)
        transition = log_normalize(logsumexp(np.log(xi), axis=0) - logsumexp(np.log(p_j_at_t), axis=0), axis=1)

        T = len(sequence)
        _, _, POS = p_j_at_t.shape

        pi = p_j_at_t[0]

        log_emission_T = np.full((vocabulary_size, POS), -np.inf)
        for t in range(T):
            # V x 1 @ 1 x POS = V x POS
            log_emission_T = np.logaddexp(
                log_emission_T,
                log_mat_mul(ohe.encode_row_log(sequence[t]).T, np.log(p_j_at_t[t]))
            )
        emission_T = log_normalize(log_emission_T, axis=0)
        return pi, transition, emission_T


def seed_matrices(
        n_hidden: int,
        n_observed: int,
        seed: int = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """

    :param n_hidden:
    :param n_observed:
    :param seed:
    :return: pi, transition, emission
    """
    np.random.seed(seed)
    pi = np.random.rand(n_hidden)
    pi = pi / np.sum(pi)
    transition = np.random.rand(n_hidden, n_hidden)
    transition = transition / np.expand_dims(transition.sum(axis=1), axis=1)
    emission = np.random.rand(n_hidden, n_observed)
    emission = emission / np.expand_dims(emission.sum(axis=1), axis=1)
    return pi, transition, emission
