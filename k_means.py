import pickle
import torch
from sklearn.cluster import KMeans

# n_sentences = 39651
n_sentences = 10
n_clusters = 17

if __name__ == "__main__":
    tokens = []
    token_embeddings = []
    for sentence_idx in range(n_sentences):
        f = open(f"checkpoints/bert_tokens/sentence{sentence_idx}.pkl", "rb")
        sentence_tokens: list[str] = pickle.load(f)
        tokens += sentence_tokens
        f.close()
        sentence_token_embeddings = torch.load(
            f"checkpoints/bert_word_embedding/sentence{sentence_idx}.pt",
            map_location="cpu",
            weights_only=True)
        token_embeddings += sentence_token_embeddings

    # is cosine distance because it is already normalized
    token_embeddings = torch.stack(token_embeddings)
    token_embeddings = token_embeddings / token_embeddings.sum(dim=0)

    # n_tokens x 768
    kmeans = KMeans(n_clusters=n_clusters, n_init="auto", max_iter=30).fit(token_embeddings)

    f = open("checkpoints/cluster.csv", "w")
    for token, label in zip(tokens, kmeans.labels_):
        # item() to get native py int
        f.write(f"{token}, {label.item()}\n")
    f.close()