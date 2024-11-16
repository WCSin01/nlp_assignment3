# v_measure:
# a permutation of the class or cluster label values won’t change the score value in any way.
# symmetric: switching label_true with label_pred will return the same score value.
import numpy as np
from hmmlearn import hmm
from sklearn.metrics.cluster import v_measure_score

from forward_backward_hmm import HMMParameters
from functions import pickle_load
from process_conllu import ConlluDataset, OneHot
from utils import calculate_variation_of_information

if __name__ == "__main__":
    # select POS type
    parameters: HMMParameters = pickle_load("checkpoints/forward_backward_xpos/epoch2.pkl")
    dataset: ConlluDataset = pickle_load("checkpoints/dataset.pkl")
    print("data loaded")

    # select POS type
    ohe = OneHot(dataset.xpos_set)
    _, V = parameters.emission.shape

    print("running viterbi...")
    pred_hidden_seqs = []
    for seq_idx, obs_seq in enumerate(dataset.sequences):
        T = len(obs_seq)
        encoded_obs_seq = np.zeros((T, 1), dtype=np.int16)
        for t in range(T):
            encoded_obs_seq[t, 0] = dataset.ohe.get_index(obs_seq[t])

        # select POS type
        model = hmm.CategoricalHMM(n_components=len(dataset.xpos_set))
        model.startprob_ = parameters.pi[0]
        model.transmat_ = parameters.transition
        model.emissionprob_ = parameters.emission

        pred_hidden_seq = model.predict(encoded_obs_seq)
        pred_hidden_seqs.append(pred_hidden_seq)

    print("writing predicted sequences...")
    # select POS type
    f = open("results/viterbi_xpos.csv", "w")
    f.write("sequence,word,hidden state")
    for seq_idx, (obs_seq, hidden_seq) in enumerate(zip(dataset.sequences, pred_hidden_seqs)):
        f.write(f'{seq_idx},"{obs_seq}",{hidden_seq}')
    f.close()

    print("encoding true hidden sequences...")
    encoded_true_hidden_seqs: list[np.ndarray] = []
    # select POS type
    for seq_idx, true_hidden_seq in enumerate(dataset.xpos):
        T = len(true_hidden_seq)
        encoded_true_hidden_seq = np.zeros(T, dtype=np.int16)
        for t in range(T):
            encoded_true_hidden_seq[t] = ohe.get_index(true_hidden_seq[t])
        encoded_true_hidden_seqs.append(encoded_true_hidden_seq)

    print("evaluting...")
    # select POS type
    f = open("results/eval_hmm_xpos.csv", "w")
    f.write("v measure,voi,normalised voi\n")
    for pred_seq, true_seq in zip(pred_hidden_seqs, encoded_true_hidden_seqs):
      v_measure = v_measure_score(pred_seq, true_seq)
      variation_of_information, norm_voi = \
          calculate_variation_of_information(pred_seq, true_seq)
      f.write(f"{v_measure},{variation_of_information},{norm_voi}\n")
    f.close()