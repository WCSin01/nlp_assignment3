import numpy as np
import pandas as pd
from sklearn.metrics import v_measure_score
from functions import flatten, pickle_load

from process_conllu import ConlluDataset, OneHot
from utils import calculate_variation_of_information

seed_num = 0

if __name__ == "__main__":
    # set POS type
    df = pd.read_csv("results/k_means_upos.csv", delimiter=", ")
    dataset: ConlluDataset = pickle_load("checkpoints/dataset.pkl")
    print("data loaded")

    # set POS type
    pos_ohe = OneHot(dataset.upos_set)

    bert_cluster = df[df["seed"] == seed_num]["cluster"].to_numpy()

    cum_sentence_idxs = np.zeros(len(dataset.sequences)+1, dtype=np.int16)
    for sentence_idx, sentence in enumerate(dataset.sequences):
        cum_sentence_idxs[sentence_idx+1] = len(sentence)
    cum_sentence_idxs = np.cumsum(cum_sentence_idxs)

    # set POS type
    f = open("results/eval_bert_upos.csv", "w")
    f.write("v measure,voi,normalised voi\n")

    # set POS type
    for sentence_idx, (sentence_pos, start, end) in enumerate(zip(
        dataset.upos, cum_sentence_idxs[:-1], cum_sentence_idxs[1:])):
        if sentence_idx % 5000 == 0:
            print(f"evaluating seq {sentence_idx}/{len(dataset.sequences)}")
        sentence_pos_encoded = [pos_ohe.get_index(pos) for pos in sentence_pos]
        word_embeddings_for_sentence = bert_cluster[start:end]
        v_measure = v_measure_score(word_embeddings_for_sentence, sentence_pos_encoded)
        variation_of_information, norm_voi = \
              calculate_variation_of_information(word_embeddings_for_sentence, sentence_pos_encoded)
        f.write(f"{v_measure},{variation_of_information},{norm_voi}\n")
    f.close()
        


