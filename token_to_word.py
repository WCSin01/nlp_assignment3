import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from functions import pickle_dump, pickle_load

device = 'cpu'
model_value = "bert-base-uncased"

if __name__ == "__main__":
    sentences = pickle_load("checkpoints/sentences.pkl")
    tokenizer = BertTokenizer.from_pretrained(model_value)
    model = BertModel.from_pretrained(model_value, output_hidden_states=True).to(device)
    cum_token_count_by_word: list[np.ndarray] = []
    # tokenizer tokenizes a word the same way whether on its own or in a sentence
    n_sentences = len(sentences)
    for sentence_idx, word_list in enumerate(sentences):
        token_count_by_word_in_sentence = np.zeros(len(word_list)+1, dtype=np.int16)
        for word_idx, word in enumerate(word_list):
            tokenized_text: list[str] = tokenizer.tokenize(word)
            token_count_by_word_in_sentence[word_idx+1] = len(tokenized_text)
        token_count_by_word_in_sentence = np.cumsum(token_count_by_word_in_sentence)
        cum_token_count_by_word.append(token_count_by_word_in_sentence)
        
        if sentence_idx % 5000 == 0:
            print(f"calculated token count for {sentence_idx+1}/{len(sentences)} sentences")

    word_embeddings: list[np.ndarray] = []
    token_embeddings: list[torch.Tensor] = torch.load(
        f"checkpoints/bert_token_embeddings_clean.pt",
        weights_only=True)
    for sentence_idx in range(n_sentences):       
        # remove special tokens
        token_embeddings_for_sentence = token_embeddings[sentence_idx].numpy()[1:-1]
        sentence_len = cum_token_count_by_word[sentence_idx][-1]
        word_embeddings_for_sentence = np.zeros((sentence_len, token_embeddings_for_sentence.shape[1]))
        for word_idx, (start, end) in enumerate(zip(
            cum_token_count_by_word[sentence_idx][:-1],
            cum_token_count_by_word[sentence_idx][1:])):
            word_embeddings_for_sentence[word_idx] = np.sum(token_embeddings_for_sentence[start:end], axis=0)
        word_embeddings.append(word_embeddings_for_sentence)

        if sentence_idx % 2000 == 0:
          print(f"remapped token to word for {sentence_idx+1}/{len(sentences)} sentences")

    pickle_dump(word_embeddings, "checkpoints/word_embeddings.pkl")
