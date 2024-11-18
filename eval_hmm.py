# v_measure:
# a permutation of the class or cluster label values won’t change the score value in any way.
# symmetric: switching label_true with label_pred will return the same score value.
import numpy as np
from hmmlearn import hmm
from sklearn.metrics.cluster import v_measure_score

from forward_backward_hmm import HMMParameters
from functions import pickle_dump, pickle_load
from process_conllu import ConlluDataset, OneHot
from utils import calculate_variation_of_information

pos = "xpos"

parameters_file_path = f"checkpoints/forward_backward_{pos}/epoch5.pkl"
eval_file_path = f"results/eval_hmm_{pos}.csv"

if __name__ == "__main__":
    parameters: HMMParameters = pickle_load(parameters_file_path)
    dataset: ConlluDataset = pickle_load("checkpoints/dataset.pkl")

    if pos == "upos":
        pos_set = dataset.upos_set
        pos_data = dataset.upos
    elif pos == "xpos":
        pos_set = dataset.xpos_set
        pos_data = dataset.xpos

    pos_ohe = OneHot(pos_set)
    _, V = parameters.emission.shape

    print("running viterbi...")
    pred_hidden_seqs = []
    for i, obs_seq in enumerate(dataset.sequences):
        T = len(obs_seq)
        encoded_obs_seq = np.zeros((T, 1), dtype=np.int16)
        for t in range(T):
            encoded_obs_seq[t, 0] = dataset.ohe.get_index(obs_seq[t])

        model = hmm.CategoricalHMM(n_components=len(pos_set))
        model.startprob_ = parameters.pi[0]
        model.transmat_ = parameters.transition
        model.emissionprob_ = parameters.emission

        pred_hidden_seq = model.predict(encoded_obs_seq)
        pred_hidden_seqs.append(pred_hidden_seq)

    print("saving viterbi...")
    pickle_dump(pred_hidden_seqs, "checkpoints/viterbi_xpos.pkl")

    print("encoding true sequences...")
    encoded_true_hidden_seqs: list[np.ndarray] = []
    for i, true_hidden_seq in enumerate(pos_data):
        T = len(true_hidden_seq)
        encoded_true_hidden_seq = np.zeros(T, dtype=np.int16)
        for t in range(T):
            encoded_true_hidden_seq[t] = pos_ohe.get_index(true_hidden_seq[t])
        encoded_true_hidden_seqs.append(encoded_true_hidden_seq)

    print("evaluating...")
    f = open("results/eval_hmm.csv", "w")
    f.write("v measure,voi,normalised voi\n")
    for pred_seq, true_seq in zip(pred_hidden_seqs, encoded_true_hidden_seqs):
      v_measure = v_measure_score(pred_seq, true_seq)
      variation_of_information, norm_voi = \
          calculate_variation_of_information(pred_seq, true_seq)
    f.write(f"{v_measure},{variation_of_information},{norm_voi}\n")
    f.close()