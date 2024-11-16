import numpy as np
import pandas as pd
from sklearn.metrics import v_measure_score
from functions import flatten, pickle_load

from process_conllu import ConlluDataset, OneHot
from utils import calculate_variation_of_information

n_seeds = 5

if __name__ == "__main__":
    df = pd.read_csv("results/k_means_xpos.csv", delimiter=", ")
    print("clusters loaded")

    dataset: ConlluDataset = pickle_load("checkpoints/dataset.pkl")
    # set POS type
    pos_ohe = OneHot(dataset.xpos_set)
    pos_for_words = flatten(dataset.xpos)
    
    pos_encoded = np.zeros(len(pos_for_words), dtype=np.int16)
    for i, pos in enumerate(pos_for_words):
        pos_encoded[i] = pos_ohe.get_index(pos)
    print("pos encoded")

    f = open("results/eval_bert_xpos.csv", "w")
    f.write("seed,v measure,voi,normalised voi\n")
    for i in range(n_seeds):
        print(f"evaluating seed {i}...")
        bert_cluster = df[df["seed"] == i]["cluster"].to_numpy()

        v_measure = v_measure_score(bert_cluster, pos_encoded)
        variation_of_information, norm_voi = \
              calculate_variation_of_information(bert_cluster, pos_encoded)
        f.write(f"{i},{v_measure},{variation_of_information},{norm_voi}\n")
    f.close()
        


