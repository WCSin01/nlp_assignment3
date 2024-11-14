import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from functions import pickle_dump, pickle_load

n_sentences = 10
device = 'cpu'
model_value = "bert-base-uncased"

if __name__ == "__main__":
    sentences = pickle_load("checkpoints/sentences.pkl")
    tokenizer = BertTokenizer.from_pretrained(model_value)
    model = BertModel.from_pretrained(model_value, output_hidden_states=True).to(device)
    cum_token_count_by_word: list[np.ndarray] = []
    # tokenizer tokenizes a word the same way whether on its own or in a sentence
    for sentence_idx, word_list in enumerate(sentences):
        token_count_by_word_in_sentence = np.zeros(len(word_list)+1, dtype=np.int16)
        for word_idx, word in enumerate(word_list):
            tokenized_text: list[str] = tokenizer.tokenize(word)
            token_count_by_word_in_sentence[word_idx+1] = len(tokenized_text)
        token_count_by_word_in_sentence = np.cumsum(token_count_by_word_in_sentence)
        cum_token_count_by_word.append(token_count_by_word_in_sentence)

    word_embeddings: list[np.ndarray] = []
    for sentence_idx in range(n_sentences):
        token_embeddings: torch.Tensor = torch.load(
                f"checkpoints/bert_word_embedding/sentence{sentence_idx}.pt",
                map_location="cpu",
                weights_only=True)
        sentence_len = len(token_count_by_word_in_sentence[sentence_idx])
        word_embeddings_for_sentence = torch.empty(sentence_len, token_embeddings.shape[1])
        for word_idx, (start, end) in enumerate(zip(
            token_count_by_word_in_sentence[sentence_idx][:-1],
            token_count_by_word_in_sentence[sentence_idx][1:])):
            word_embeddings_for_sentence[word_idx] = token_embeddings[start:end].sum(dim=0).divide(sentence_len)
        word_embeddings.append(word_embeddings_for_sentence.numpy())
    pickle_dump(word_embeddings, "checkpoints/word_embeddings.pkl")
