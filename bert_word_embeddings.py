import torch
from transformers import BertTokenizer, BertModel

from process_data import process_conllu

model_value = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_value)
model = BertModel.from_pretrained(model_value, output_hidden_states=True)
dataset = process_conllu("ptb-train.conllu")
for sentence in dataset.sentences:
    # token_tensors, segment_tensors
    token_tensors, segment_tensors = tokenizer.tokenize(sentence.join(" "), return_tensors='pt')
    print(token_tensors)

    model.eval()
    with torch.no_grad():
        output = model(token_tensors, segment_tensors)
        # for output_hidden_states=True, index 2 is hidden_state
        # dimensions: 13 layers (input + 12 hidden) x n_batch (1) xn_token x n_feature (768)
        # first dim is a list
        hidden_state = output[2]
        # to one tensor
        hidden_state = torch.stack(hidden_state)
        # 13 x n_token x 768

        torch.squeeze(token)

    break
