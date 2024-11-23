import numpy as np
import pandas as pd
from sklearn.metrics import v_measure_score
from functions import flatten, pickle_load

from process_conllu import ConlluDataset, OneHot
from utils import calculate_variation_of_information

pos = "xpos"
n_seeds = 5

k_means_file_path = f"results/k_means_{pos}.csv"
eval_file_path = f"results/eval_bert_{pos}.csv"

if __name__ == "__main__":
    df = pd.read_csv(k_means_file_path)
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
    flattened_pos = [pos_ohe.get_index(pos) for pos in flatten(pos_data)]
    print("pos flattened")

    f = open(eval_file_path, "w")
    f.write("seed,v measure,voi,normalised voi\n")
    for seed in range(n_seeds):
        bert_cluster = df[df["seed"] == seed]["cluster"].to_numpy()
        v_measure = v_measure_score(flattened_pos, bert_cluster)
        variation_of_information, norm_voi = \
              calculate_variation_of_information(flattened_pos, bert_cluster)
        f.write(f"{seed},{v_measure},{variation_of_information},{norm_voi}\n")
    f.close()


