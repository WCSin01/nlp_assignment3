# Installation

Install pytorch: https://pytorch.org/get-started/locally/

```
pip install -r requirements.txt
```

Build directories

```
mkdir results
mkdir checkpoints
cd checkpoints
mkdir bert_token_embeddings
mkdir forward_backward_upos
mkdir forward_backward_xpos
```

# Pipeline

Data analysis and preprocessing

```
dataset_summary.ipynb
run_process_conllu.py
```

Main code is imported from `process_conllu.py`

Subtask 1

```
run_forward_backward_hmm.py
```

Main code is imported from `forward_backward_hmm.py`

Subtask 2

```
bert_token_embeddings.py
token_to_word.py
k_means.py
```

Subtask 3

```
eval_hmm.py
eval_bert.py
eval_plt.ipynb
```

# Expected runtime duration and hardware requirements

Hardware used: i5-10300H

* forward backward: 4 min per epoch
* bert token embeddings (GPU): 8 min, 7GB of disk to unzip, 4GB for only unzipped
* bert token embeddings (CPU): 30 min
* k means clustering: 4 min per seed of 50 iterations, 16GB of memory may not be sufficient
