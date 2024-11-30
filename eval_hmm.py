import numpy as np
from hmmlearn import hmm
from sklearn.metrics.cluster import v_measure_score

from forward_backward_hmm import HMMParameters
from functions import flatten, pickle_load, pickle_dump
from process_conllu import ConlluDataset, OneHot
from utils import calculate_variation_of_information

pos = "upos"
filter_count = 1

epochs = [1] + list(range(20, 210, 20))
eval_file_path = f"results/eval_hmm_{pos}_{filter_count}.csv"
viterbi_file_path = f"results/viterbi_{pos}_{filter_count}.pkl"

if __name__ == "__main__":
    dataset: ConlluDataset = pickle_load("checkpoints/dataset.pkl")
    print("data loaded")

    if pos == "upos":
        pos_set = dataset.upos_set
        pos_data = dataset.upos
    elif pos == "xpos":
        pos_set = dataset.xpos_set
        pos_data = dataset.xpos
    else:
        raise NotImplementedError

    pos_ohe = OneHot(pos_set)

    f = open(eval_file_path, "w")
    f.write("epoch,v measure,voi,normalised voi\n")
    for epoch in epochs:
        parameters_file_path = f"checkpoints/forward_backward_{pos}/{filter_count}/epoch{epoch}.pkl"
        parameters: HMMParameters = pickle_load(parameters_file_path)

        pred_hidden_seqs = []
        for i, obs_seq in enumerate(dataset.sequences):
            T = len(obs_seq)
            encoded_obs_seq = np.zeros((T, 1), dtype=np.int16)
            for t in range(T):
                encoded_obs_seq[t, 0] = dataset.ohe.get_index(obs_seq[t])

            model = hmm.CategoricalHMM(n_components=len(pos_set))
            model.startprob_ = parameters.pi
            model.transmat_ = parameters.transition
            model.emissionprob_ = parameters.emission

            pred_hidden_seq = model.predict(encoded_obs_seq)
            pred_hidden_seqs.append(pred_hidden_seq)

        if epoch == epochs[-1]:
            pickle_dump(pred_hidden_seqs, )
        print("viterbi done")

        encoded_true_hidden_seqs: list[np.ndarray] = []
        for i, true_hidden_seq in enumerate(pos_data):
            T = len(true_hidden_seq)
            encoded_true_hidden_seq = np.zeros(T, dtype=np.int16)
            for t in range(T):
                encoded_true_hidden_seq[t] = pos_ohe.get_index(true_hidden_seq[t])
            encoded_true_hidden_seqs.append(encoded_true_hidden_seq)

        print("true sequence encoded")

        flattened_pred = flatten(pred_hidden_seqs)
        flattened_true = flatten(encoded_true_hidden_seqs)
        v_measure = v_measure_score(flattened_true, flattened_pred)
        variation_of_information, norm_voi = calculate_variation_of_information(flattened_true, flattened_pred)
        f.write(f"{epoch},{v_measure},{variation_of_information},{norm_voi}\n")
        print(f"epoch {epoch} evaluated")
    f.close()