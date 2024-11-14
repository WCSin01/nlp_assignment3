import numpy as np
import pandas as pd
from sklearn.metrics import v_measure_score
from functions import flatten, pickle_load

from process_conllu import ConlluDataset, OneHot
from utils import calculate_variation_of_information

if __name__ == "__main__":
    df = pd.read_csv("results/k_means.csv")
    bert_cluster = df["cluster"].to_numpy()

    dataset: ConlluDataset = pickle_load("checkpoints/dataset.pkl")
    upos_ohe = OneHot(dataset.upos_set)
    upos_list = flatten(dataset.upos)
    upos_encoded = np.zeros(len(upos_list))
    for i, upos in enumerate(upos_list):
        upos_encoded[i] = upos_ohe.get_index(upos)

    f = open("results/eval_bert.csv", "w")
    v_measure = v_measure_score(bert_cluster, upos_encoded)
    variation_of_information, norm_voi = \
          calculate_variation_of_information(bert_cluster, upos_encoded)
    f.write("v measure, voi, normalised voi\n")
    f.write(f"{v_measure}, {variation_of_information}, {norm_voi}\n")
    f.close()
    


