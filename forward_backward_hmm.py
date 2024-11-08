from dataclasses import dataclass
import numpy as np
from scipy.special import logsumexp
from numeric import log_normalize, log_mat_mul
from process_data import ConlluDataset


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


def forward_backward(
        dataset: ConlluDataset,
        pi: np.ndarray,
        transition: np.ndarray,
        emission: np.ndarray,
        max_iter: int) -> ForwardBackwardOutput:
    """

    :param observed_sequences:
    :param pi: initial probability distribution. |POS|
    :param transition: transition matrix. axis=0: current state, axis=1: next state. |POS|^2
    :param emission: emission matrix. axis=0: hidden state, axis=1: observed state. |POS| x |vocabulary|
    :param max_iter: maximum iteration to stop if no convergence.
    :return: pi, transition, emission, if_converged
    """
    POS, V = emission.shape
    assert POS == pi.shape[0]
    assert (POS, POS) == transition.shape
    assert np.allclose(pi.sum(), 1)
    assert np.allclose(transition.sum(axis=1), 1)
    assert np.allclose(emission.sum(axis=1), 1)

    # np.log(0) = -np.inf
    with np.errstate(divide='ignore'):
        # increase dimension for mat_mul
        log_pi = np.expand_dims(np.log(pi), axis=0)
        log_transition = np.log(transition)
        log_emission_T = np.log(emission).T

        # one outer loop is one iteration through the whole dataset
        for i in range(max_iter):
            for j, sentence in enumerate(dataset.data):
                sequence = np.zeros((len(sentence), 1, dataset.vocabulary_size))
                for k, word in enumerate(sentence):
                    sequence[k] = np.array([dataset.ohe.encode(word)])
                # increase dimension for mat_mul
                log_observed = np.log(sequence)
                p_j_at_t, log_xi = forward_backward_expect(log_observed, log_pi, log_transition, log_emission_T)
                new_pi, new_transition, new_emission_T = forward_backward_max(p_j_at_t, log_xi, log_observed)

                print(f"epoch: {i+1}, sequence #: {j+1}/{dataset.dataset_size}")

            if np.allclose(np.exp(log_transition), new_transition) and np.allclose(np.exp(log_emission_T), new_emission_T):
                return ForwardBackwardOutput(new_pi, new_transition, new_emission_T.T, True)
            else:
                log_transition = np.log(new_transition)
                log_emission_T = np.log(new_emission_T)

        return ForwardBackwardOutput(new_pi, new_transition, new_emission_T.T, True)


def log_forward(
        log_observed: np.ndarray,
        log_pi: np.ndarray,
        log_transition: np.ndarray,
        log_emission_T: np.ndarray) -> np.ndarray:
    """

    :param log_observed:
    :param log_pi:
    :param log_transition:
    :param log_emission_T:
    :return: log a_t(j) = log P(o_1 o_2 ... o_t, q_t=j|lambda). Tx1x|POS|
    """
    T, _, V = log_observed.shape
    _, POS = log_pi.shape

    # initialize
    # length x POS
    log_p_forward = np.full((T, 1, POS), -np.inf)

    # 1 x V @ V x POS = 1 x POS
    # 1 x POS * 1 x POS
    log_p_forward[0] = log_pi + log_mat_mul(log_observed[0], log_emission_T)

    # recursion
    for t in range(1, T):
        # 1 x POS @ POS x POS = 1 x POS
        log_p_forward[t] = log_mat_mul(log_p_forward[t - 1], log_transition) + \
                           log_mat_mul(log_observed[t], log_emission_T)

    return log_p_forward


def log_backward(
        log_observed: np.ndarray,
        log_pi: np.ndarray,
        log_transition: np.ndarray,
        log_emission_T: np.ndarray) -> np.ndarray:
    """

    :param log_observed:
    :param log_pi:
    :param log_transition:
    :param log_emission_T:
    :return: log b_t(i) = log P(o_t+1 o_t+2 ... o_T|q_t = i, lambda). Tx1x|POS|
    """
    T, _, V = log_observed.shape
    _, POS = log_pi.shape

    # initialize
    log_p_backward = np.full((T, 1, POS), -np.inf)
    log_p_backward[-1] = np.zeros((1, POS))  # this is not a probability distribution

    # recursion
    # from T-2 to 0 inclusive
    for t in range(T - 2, -1, -1):
        # 1 x V @ V x POS = 1 x POS
        # 1 x POS @ POS x POS = 1 x POS
        log_p_backward[t] = log_mat_mul(log_mat_mul(log_observed[t + 1], log_emission_T), log_transition) + \
                            log_p_backward[t + 1, :]

    return log_p_backward


def expected_state_occupancy_count(
        log_forward_mat: np.ndarray,
        log_backward_mat: np.ndarray) -> np.ndarray:
    """

    :param log_forward_mat:
    :param log_backward_mat:
    :param norm:
    :return: P(q_t=j|O, lambda). T x 1 x |POS|
    """
    p_j_at_t = log_normalize(log_forward_mat + log_backward_mat, axis=2)
    return p_j_at_t


def expected_state_transition_count(
        log_forward_mat,
        log_backward_mat,
        log_transition,
        log_emission_T,
        log_observed) -> np.ndarray:
    """

    :param log_forward_mat:
    :param log_backward_mat:
    :param log_transition:
    :param log_emission_T:
    :param log_observed:
    :param norm:
    :return: xi_t(ij).
        P(q_t=i, q_{t+1}=j|O, lambda).
        the probability at state i at time t and state j at time t+1.
        (T-1) x |POS| x |POS|
    """
    T, _, POS = log_forward_mat.shape
    log_xi = np.zeros((T - 1, POS, POS))
    log_transition_T = log_transition.T

    for t in range(T - 1):
        # 1 x POS automatic broadcast to POS x POS
        # POS x POS * POS x POS
        log_xi[t] = log_forward_mat[t] + log_transition_T + \
                    log_mat_mul(log_observed[t + 1], log_emission_T) + log_backward_mat[t + 1].T
    xi = log_normalize(log_xi, axis=2)
    return xi


def forward_backward_expect(
        log_observed: np.ndarray,
        log_pi: np.ndarray,
        log_transition: np.ndarray,
        log_emission_T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """

    :param log_observed:
    :param log_pi:
    :param log_transition:
    :param log_emission_T:
    :return: p_j_at_t, log_xi
    """
    log_forward_mat = log_forward(log_observed, log_pi, log_transition, log_emission_T)
    log_backward_mat = log_backward(log_observed, log_pi, log_transition, log_emission_T)
    # (T-1) x 1 x 1
    norm = np.expand_dims(logsumexp(log_forward_mat + log_backward_mat, axis=2), axis=2)
    p_j_at_t = expected_state_occupancy_count(log_forward_mat, log_backward_mat)
    log_xi = expected_state_transition_count(
        log_forward_mat,
        log_backward_mat,
        log_transition,
        log_emission_T,
        log_observed)
    return p_j_at_t, log_xi


def forward_backward_max(
        p_j_at_t: np.ndarray,
        log_xi: np.ndarray,
        log_observed: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """

    :param p_j_at_t:
    :param log_xi:
    :param log_observed:
    :return: pi, transition, emission_T
    """
    log_transition = logsumexp(log_xi, axis=0)
    transition = log_normalize(log_transition, axis=1)

    T, _, V = log_observed.shape
    _, _, POS = p_j_at_t.shape

    pi = p_j_at_t[0]
    assert np.sum(pi) == 1

    log_emission_T = np.full((V, POS), -np.inf)
    for t in range(T):
        # V x 1 @ 1 x POS = V x POS
        log_emission_T = np.logaddexp(
            log_emission_T,
            log_mat_mul(log_observed[t].T, np.log(p_j_at_t[t]))
        )
    emission_T = log_normalize(log_emission_T, axis=0)

    return pi, transition, emission_T


def seed_matrices(n_hidden: int, n_observed: int, seed: int = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
