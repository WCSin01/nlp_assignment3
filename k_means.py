import torch
import numpy as np
from sklearn.cluster import KMeans

from functions import flatten, pickle_load
from process_conllu import ConlluDataset

# n_sentences = 39651
# n_clusters = 17

if __name__ == "__main__":
    dataset: ConlluDataset = pickle_load(f"checkpoints/dataset.pkl")
    word_embeddings: list[np.ndarray] = pickle_load("checkpoints/word_embeddings.pkl")
    print("data loaded")

    # is cosine distance because it is already normalized
    flat_word_embeddings = np.concatenate(word_embeddings)
    print(flat_word_embeddings.shape)
    print(np.max(flat_word_embeddings))
    print(np.min(flat_word_embeddings))
    np.savetxt("sum.txt", np.sum(flat_word_embeddings, axis=1))
    exit()

    # n_tokens x 768
    kmeans = KMeans(n_clusters=len(dataset.upos_set), n_init="auto", max_iter=30).fit(flat_word_embeddings)

    f = open("results/k_means.csv", "w")
    f.write(f"word, cluster\n")
    for token, label in zip(flatten(dataset.sequences), kmeans.labels_):
        # item() to get native py int
        f.write(f"{token}, {label.item()}\n")
    f.close()