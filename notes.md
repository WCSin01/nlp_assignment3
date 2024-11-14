# conllu format

```
id: 1
form: In
lemma: _
upos: ADP
xpos: IN
feats: None
head: 45
deprel: prep
deps: None
misc: None
```

```
    :param pi: initial probability distribution. |POS|
    :param transition: transition matrix. axis=0: current state, axis=1: next state. |POS|^2
    :param emission: emission matrix. axis=0: hidden state, axis=1: observed state. |POS| x |vocabulary|
```

```
    # T, V = observed.shape
    # POS = pi.shape[0]
    # assert (POS, POS) == transition.shape
    # assert (POS, V) == emission.shape
    # assert np.allclose(pi.sum(), 1)
    # assert np.allclose(transition.sum(axis=1), 1)
    # assert np.allclose(emission.sum(axis=1), 1)
```

# TODO

rename bert_token_embedding, sentence_ls

# Misc

bert <5 hours

100 iterations, kmeans 6 hours