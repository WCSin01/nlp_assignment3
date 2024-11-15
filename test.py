import torch
from forward_backward_hmm import HMMParameters
from functions import pickle_dump, pickle_load
import numpy as np

if __name__ == "__main__":
    pass

    # params: HMMParameters = pickle_load("checkpoints/forward_backward_saved/epoch5.pkl")

    # assert np.isclose(np.sum(params.pi.flatten()), 1)
    # assert np.isclose(np.all(np.sum(params.transition, axis=1)), 1)
    # assert np.isclose(np.all(np.sum(params.emission, axis=1)), 1)
    # assert not np.any(np.isnan(params.pi.flatten()))
    # assert not np.any(np.isnan(params.transition))
    # assert not np.any(np.isnan(params.emission))