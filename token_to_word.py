from time import sleep
import numpy as np
from transformers import BertTokenizer
from functions import pickle_dump, pickle_load

model_value = "bert-base-uncased"
file_path = "/home/wcs26/rds/hpc-work/nlp/bert_token_embeddings"

if __name__ == "__main__":
    sentences = pickle_load("checkpoints/sentences.pkl")
    tokenizer = BertTokenizer.from_pretrained(model_value)
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
    for sentence_idx in range(n_sentences):
        # remove special tokens
        token_embeddings_for_sentence = np.load(f"{file_path}/{sentence_idx}.npy")[1:-1]
        word_embeddings_for_sentence = np.zeros((len(sentences[sentence_idx]), token_embeddings_for_sentence.shape[1]))
        for word_idx, (start, end) in enumerate(zip(
            cum_token_count_by_word[sentence_idx][:-1],
            cum_token_count_by_word[sentence_idx][1:])):
            word_embeddings_for_sentence[word_idx] = np.sum(token_embeddings_for_sentence[start:end], axis=0)
        word_embeddings.append(word_embeddings_for_sentence)

        if sentence_idx % 2000 == 0:
          print(f"remapped token to word for {sentence_idx+1}/{len(sentences)} sentences")

    word_embeddings = np.concatenate(word_embeddings)
    np.save("checkpoints/word_embeddings", word_embeddings)
