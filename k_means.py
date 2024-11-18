import torch
import numpy as np
from sklearn.cluster import KMeans

from functions import flatten, pickle_load
from numeric import masked_normalize
from process_conllu import ConlluDataset

pos = "xpos"

if __name__ == "__main__":
    if pos == "upos":
        n_clusters = 17
    elif pos == "xpos":
        n_clusters = 45
    else:
        raise NotImplementedError

    sentences: list[list[str]] = pickle_load(f"checkpoints/sentences.pkl")
    word_embeddings: np.ndarray = np.load("checkpoints/word_embeddings.npy")
    print("data loaded")

    word_embeddings = masked_normalize(word_embeddings, axis=1)
    print("embeddings normalized")

    f = open(f"results/k_means_{pos}.csv", "w")
    f.write(f"word,seed,cluster\n")
    for i in range(1):
        # n_tokens x 768
        kmeans = KMeans(n_clusters=n_clusters, n_init="auto", max_iter=50).fit(word_embeddings)
        print(f"clustered seed {i}")

        for token, label in zip(flatten(sentences), kmeans.labels_):
            # item() to get native py int
            f.write(f'"{token}",{i},{label.item()}\n')
        print(f"seed {i} results written")
    f.close()
