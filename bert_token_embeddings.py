# Modified from https://mccormickml.com/2019/05/14/BERT-word-embeddings-tutorial
import pickle
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from functions import pickle_dump, pickle_load
from process_conllu import ConlluProcessor

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_value = "bert-base-uncased"

# 5 mins on GPU
if __name__ == "__main__":
    sentences = pickle_load("checkpoints/sentences.pkl")
    tokenizer = BertTokenizer.from_pretrained(model_value)
    model = BertModel.from_pretrained(model_value, output_hidden_states=True).to(device)

    token_embeddings = dict()
    for sentence_idx, word_list in enumerate(sentences):
        if sentence_idx not in {19}:
            continue

        if sentence_idx % 500 == 0:
            print(f"sentence {sentence_idx+1}/{len(sentences)}")

        sentence = " ".join(word_list)
        marked_text = f"[CLS] {sentence} [SEP]"
        tokenized_text: list[str] = tokenizer.tokenize(marked_text)

        indexed_tokens: list[int] = tokenizer.convert_tokens_to_ids(tokenized_text)
        token_tensor = torch.tensor([indexed_tokens]).to(device)
        segment_tensor = torch.ones(1, len(tokenized_text)).to(device)

        model.eval()
        with torch.no_grad():
            # 1. last hidden state (batch_size, sequence_length, hidden_size)
            # 2. pooler output of classification token (batch_size, hidden_size)
            # 3. initial embedding + hidden states (n_layer=13, batch_size=1, n_token, hidden_size=768).
            #    First dim is a list, not tensor.
            output = model(token_tensor, segment_tensor)
            hidden_state = output[2]
            hidden_state = torch.stack(hidden_state)
            # 13 x n_token x 768
            hidden_state = torch.squeeze(hidden_state, dim=1)
            # n_token x 13 x 768
            hidden_state = hidden_state.permute(1, 0, 2)

            # for best results + reduce dimensions: average last 4 layers
            token_embeddings_for_sentence = torch.empty(hidden_state.shape[0], 768)
            # exclude special markers
            for token_idx, token in enumerate(hidden_state[1:-1]):
                # token: 13 x 768
                # token embedding: n_tokens x 768
                token_embeddings_for_sentence[token_idx] =\
                    torch.stack([token[-1], token[-2], token[-3], token[-4]], dim=0).sum(dim=0).div(4)

        token_embeddings[sentence_idx] = token_embeddings_for_sentence

    torch.save(token_embeddings, "checkpoints/bert_token_embeddings_fix_corrupted.pt")