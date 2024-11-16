import torch
import numpy as np
from sklearn.cluster import KMeans

from functions import flatten, pickle_load
from numeric import masked_normalize
from process_conllu import ConlluDataset

# n_sentences = 39651
# n_clusters = 17
n_clusters = 45

if __name__ == "__main__":
    sentences: list[list[str]] = pickle_load(f"checkpoints/sentences.pkl")
    word_embeddings: np.ndarray = np.load("checkpoints/word_embeddings.npy")
    print("data loaded")

    word_embeddings = masked_normalize(word_embeddings, axis=1)

    f = open("results/k_means_xpos.csv", "w")
    f.write(f"word, seed, cluster\n")
    for i in range(5):
      print(f"clustering seed {i}...")
      # n_tokens x 768
      kmeans = KMeans(n_clusters=n_clusters, n_init="auto", max_iter=50).fit(word_embeddings)
      print("writing results...")

      for token, label in zip(flatten(sentences), kmeans.labels_):
          # item() to get native py int
          f.write(f'{token}, {i}, {label.item()}\n')
    f.close()