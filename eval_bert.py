import numpy as np
import pandas as pd
from sklearn.metrics import v_measure_score
from functions import flatten, pickle_load

from process_conllu import ConlluDataset, OneHot
from utils import calculate_variation_of_information

seed_num = 2
pos = "upos"

k_means_file_path = f"results/k_means_{pos}.csv"
eval_file_path = f"results/eval_bert_{pos}.csv"


if __name__ == "__main__":
    df = pd.read_csv(k_means_file_path, delimiter=", ")
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

    bert_cluster = df[df["seed"] == seed_num]["cluster"].to_numpy()

    cum_sentence_idxs = np.zeros(len(dataset.sequences)+1, dtype=np.int16)
    for sentence_idx, sentence in enumerate(dataset.sequences):
        cum_sentence_idxs[sentence_idx+1] = len(sentence)
    cum_sentence_idxs = np.cumsum(cum_sentence_idxs)

    f = open(eval_file_path, "w")
    f.write("v measure,voi,normalised voi\n")

    for sentence_idx, (sentence_pos, start, end) in enumerate(zip(
        pos_data, cum_sentence_idxs[:-1], cum_sentence_idxs[1:])):
        if sentence_idx % 5000 == 0:
            print(f"evaluating seq {sentence_idx}/{len(dataset.sequences)}")
        sentence_pos_encoded = [pos_ohe.get_index(pos) for pos in sentence_pos]
        word_embeddings_for_sentence = bert_cluster[start:end]
        v_measure = v_measure_score(word_embeddings_for_sentence, sentence_pos_encoded)
        variation_of_information, norm_voi = \
              calculate_variation_of_information(word_embeddings_for_sentence, sentence_pos_encoded)
        f.write(f"{v_measure},{variation_of_information},{norm_voi}\n")
    f.close()
        


