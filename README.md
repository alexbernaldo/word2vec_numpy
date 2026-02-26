# Word2Vec Skip-Gram — Pure NumPy

From-scratch implementation of **word2vec Skip-Gram with Negative Sampling (SGNS)** using only NumPy.

Given a center word, the model learns to predict surrounding context words. The resulting vectors capture semantic relationships (`king - man + woman ≈ queen`).

## What's implemented

- Text preprocessing and vocabulary building
- `(center, context)` pair generation with a sliding window
- Negative sampling with unigram distribution raised to `3/4`
- Forward pass, loss and hand-derived gradients
- Parameter updates with SGD
- `most_similar` — nearest words by cosine similarity
- `analogy` — vector analogy operations

## Dataset

The corpus (`synthetic.txt`) is a small hand-crafted dataset designed to encode three semantic relationships:

| Relationship | Examples |
|---|---|
| Royalty | `king` ↔ `queen`, `prince` ↔ `princess` |
| Gender | `man` ↔ `woman`, `boy` ↔ `girl` |
| Life stage | `boy → man`, `girl → woman`, `prince → king` |

It consists of **31 unique sentences** (930 lines total, repeated 30×) and a vocabulary of **~25 words**. Repetition is necessary to give the model enough co-occurrence signal to learn meaningful vectors from such a small vocabulary.

It is intentionally minimal — just enough to verify that the model learns the `king - man + woman ≈ queen` analogy correctly.

## Run

```bash
pip install -r requirements.txt
python word2vec.py
```

## Papers

- Mikolov et al. (2013) — [*Efficient Estimation of Word Representations in Vector Space*](https://arxiv.org/abs/1301.3781)
  
    Introduces word2vec, the CBOW and Skip-Gram architectures.

- Mikolov et al. (2013) — [*Distributed Representations of Words and Phrases and their Compositionality*](https://arxiv.org/abs/1310.4546)

    Introduces Negative Sampling (SGNS), subsampling of frequent words, and phrase vectors.
