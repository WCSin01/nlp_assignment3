from dataclasses import dataclass
from typing import Generic, TypeVar

import numpy as np
from scipy.special import logsumexp
from sklearn.metrics import mean_squared_error
from functions import pickle_dump

from numeric import log_normalize, log_mat_mul
from process_conllu import Dataset

TypeT = TypeVar("TypeT")
min_float = 1e-323


@dataclass
class HMMParameters:
    def __init__(
            self,
            pi: np.ndarray,
            transition: np.ndarray,
            emission: np.ndarray):
        """

        :param pi: initial probability distribution. |POS|
        :param transition: transition matrix. axis=0: current state, axis=1: next state. |POS|^2
        :param emission: emission matrix. axis=0: hidden state, axis=1: observed state. |POS| x |vocabulary|
        """
        assert len(pi.shape) == 1
        POS = pi.shape[0]
        assert transition.shape == (POS, POS)
        assert emission.shape[0] == POS

        self.pi = pi
        self.transition = transition
        self.emission = emission


class HMM(Generic[TypeT]):
    def __init__(
            self,
            parameters: HMMParameters,
            dataset: Dataset[TypeT],
            save_folder_path: str):
        """

        :param parameters:
        :param dataset:
        :param save_folder_path:
        """
        assert np.allclose(parameters.pi.sum(), 1)
        assert np.allclose(parameters.transition.sum(axis=1), 1)
        assert np.allclose(parameters.emission.sum(axis=1), 1)

        self.dataset = dataset
        self.save_folder_path = save_folder_path
        # np.log(0) = -np.inf
        with np.errstate(divide='ignore', invalid='raise'):
            # increase dimension for mat_mul
            self.log_pi = np.expand_dims(np.log(parameters.pi), axis=0)
            self.log_transition = np.log(parameters.transition)
            self.log_transition_T = self.log_transition.T
            self.log_emission_T = np.log(parameters.emission).T

    def update(self, log_pi, log_transition, log_emission_T):
        self.log_pi = log_pi
        self.log_transition = log_transition
        self.log_transition_T = self.log_transition.T
        self.log_emission_T = log_emission_T

    def forward_backward(self, max_iter: int):
        """

        :param max_iter: maximum iteration to stop if no convergence.
        :return: if_converged
        """
        with np.errstate(divide='ignore', invalid='raise'):
            # one outer loop is one iteration through the whole dataset
            for i in range(max_iter):
                new_log_pi = np.full(self.log_pi.shape, -np.inf)
                new_log_transition = np.full(self.log_transition.shape, -np.inf)
                new_log_emission_T = np.full(self.log_emission_T.shape, -np.inf)

                for j, sequence in enumerate(self.dataset.sequences):
                    T = len(sequence)
                    V, _ = self.log_emission_T.shape
                    indexed_seq = [self.dataset.ohe.get_index(sequence[t]) for t in range(T)]

                    log_p_j_at_t, log_xi = self.forward_backward_expect(indexed_seq)

                    new_log_pi, new_log_transition, new_log_emission_T = \
                        self.forward_backward_max(
                            indexed_seq,
                            new_log_pi,
                            new_log_transition,
                            new_log_emission_T,
                            log_p_j_at_t,
                            log_xi)

                    if j % 1000 == 0:
                        print(f"epoch: {i + 1}/{max_iter}, sequence #: {j + 1}/{len(self.dataset.sequences)}")

                new_log_pi = log_normalize(new_log_pi, axis=1)
                new_log_transition = log_normalize(new_log_transition, axis=1)
                new_log_emission_T = log_normalize(new_log_emission_T, axis=0)

                # assert np.allclose(np.sum(new_transition, axis=1), 1, atol=1.e-3)
                # assert np.allclose(np.sum(new_emission_T, axis=0), 1, atol=1.e-3)

                transition = np.exp(self.log_transition)
                new_pi = np.exp(new_log_pi)[0]
                new_transition = np.exp(new_log_transition)
                new_emission_T = np.exp(new_log_emission_T)
                print(f"transition MSE: {mean_squared_error(new_transition.flatten(), transition.flatten())}")

                pickle_dump(
                    HMMParameters(new_pi, new_transition, new_emission_T.T),
                    f"{self.save_folder_path}/epoch{i+1}.pkl")

                if (np.allclose(transition, new_transition) and
                        np.allclose(np.exp(self.log_emission_T), new_emission_T)):
                    self.update(new_log_pi, new_log_transition, new_log_emission_T)
                    return True
                else:
                    self.update(new_log_pi, new_log_transition, new_log_emission_T)

            return False

    def save_sequence_checkpoint(
            self, i, max_iter, j, log_pi_numerator, log_transition_numerator, log_emission_T_numerator):
        f = open(f"{self.save_folder_path}/checkpoint.txt", "w")
        f.write(f"i: {i}, j: {j}")
        f.close()
        np.save(f"{self.save_folder_path}/log_pi_numerator", log_pi_numerator)
        np.save(f"{self.save_folder_path}/log_transition_numerator", log_transition_numerator)
        np.save(f"{self.save_folder_path}/log_emission_T_numerator", log_emission_T_numerator)
        print(f"epoch: {i + 1}/{max_iter}, sequence #: {j + 1}/{len(self.dataset.sequences)}")

    def log_forward(self, indexed_seq: list[int]) -> np.ndarray:
        """

        :param indexed_seq:
        :return: log a_t(j) = log P(o_1 o_2 ... o_t, q_t=j|lambda). Tx1x|POS|
        """
        with np.errstate(divide='ignore', invalid='raise'):
            T = len(indexed_seq)
            _, POS = self.log_pi.shape

            # initialize
            # length x POS
            log_p_forward = np.full((T, 1, POS), -np.inf)

            # 1 x POS * 1 x POS
            log_p_forward[0] = self.log_pi + self.log_emission_T[indexed_seq[0]]

            # recursion
            for t in range(1, T):
                # 1 x POS @ POS x POS = 1 x POS
                log_p_forward[t] = log_mat_mul(log_p_forward[t - 1], self.log_transition) + \
                                   self.log_emission_T[indexed_seq[t]]
            return log_p_forward

    def log_backward(self, indexed_seq: list[int]) -> np.ndarray:
        """

        :param indexed_seq:
        :return: log b_t(i) = log P(o_t+1 o_t+2 ... o_T|q_t = i, lambda). Tx1x|POS|
        """
        T = len(indexed_seq)
        _, POS = self.log_pi.shape

        # initialize
        log_p_backward = np.full((T, 1, POS), -np.inf)
        log_p_backward[-1] = np.zeros((1, POS))  # this is not a probability distribution

        # recursion
        # from T-2 to 0 inclusive
        for t in range(T - 2, -1, -1):
            # 1 x POS @ POS x POS = 1 x POS
            log_p_backward[t] = log_mat_mul(
                self.log_emission_T[indexed_seq[t + 1]] + log_p_backward[t + 1],
                self.log_transition_T
            )
        return log_p_backward

    def expected_state_occupancy_count(
            self,
            log_p_forward: np.ndarray,
            log_p_backward: np.ndarray) -> np.ndarray:
        """

        :param log_p_forward:
        :param log_p_backward:
        :return: log gamma. log P(q_t=j|O, lambda). T x 1 x |POS|
        """
        log_p_j_at_t = log_normalize(log_p_forward + log_p_backward, axis=2)
        return log_p_j_at_t

    def expected_state_transition_count(
            self,
            indexed_seq: list[int],
            log_p_forward: np.ndarray,
            log_p_backward: np.ndarray) -> np.ndarray:
        """

        :param encoded_sequence:
        :param log_p_forward:
        :param log_p_backward:
        :return: log xi_t(ij).
            log P(q_t=i, q_{t+1}=j|O, lambda).
            the probability at state i at time t and state j at time t+1.
            (T-1) x |POS| x |POS|
        """
        T, _, POS = log_p_forward.shape
        log_xi = np.zeros((T - 1, POS, POS))

        for t in range(T - 1):
            # axis 0: from state, axis 1: to state
            # POS x 1 * POS x POS * 1 x POS * 1 x POS
            log_xi[t] = log_p_forward[t].T + self.log_transition + \
                        self.log_emission_T[indexed_seq[t + 1]] + \
                        log_p_backward[t + 1]

        # normalize s.t. every time unit sums to 1
        log_xi = log_normalize(log_xi, axis=(1, 2))
        return log_xi

    def forward_backward_expect(self, indexed_seq: list[int]) -> tuple[np.ndarray, np.ndarray]:
        """

        :param indexed_seq:
        :return: log_p_j_at_t, log_xi
        """
        log_p_forward = self.log_forward(indexed_seq)
        log_p_backward = self.log_backward(indexed_seq)
        # p_o_lambda: P(O|lambda)
        # p_o_lambda = np.sum(log_forward_mat[-1])
        # (T-1) x 1 x 1
        log_p_j_at_t = self.expected_state_occupancy_count(log_p_forward, log_p_backward)
        log_xi = self.expected_state_transition_count(
            indexed_seq, log_p_forward, log_p_backward)
        return log_p_j_at_t, log_xi

    def forward_backward_max(
            self,
            indexed_seq: list[int],
            log_pi: np.ndarray,
            log_transition: np.ndarray,
            log_emission_T: np.ndarray,
            log_p_j_at_t: np.ndarray,
            log_xi: np.ndarray):
        """

        :param encoded_sequence:
        :param log_pi:
        :param log_transition:
        :param log_emission_T:
        :param log_p_j_at_t: T x 1 x |POS|
        :param log_xi:
        :return: log_pi, log_transition, log_emission_T
        """
        log_p_j_at_t = np.squeeze(log_p_j_at_t, axis=1)

        log_pi = np.logaddexp(log_pi, log_p_j_at_t[0])
        log_transition = np.logaddexp(log_transition, logsumexp(log_xi, axis=0))

        # V x POS -> T x POS, T x POS
        log_emission_T[indexed_seq] = np.logaddexp(log_emission_T[indexed_seq], log_p_j_at_t)
        return log_pi, log_transition, log_emission_T


def seed_matrices(
        n_hidden: int,
        n_observed: int,
        seed: int = None):
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
    return HMMParameters(pi, transition, emission)
