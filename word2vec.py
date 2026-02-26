import numpy as np
import re
import time
from collections import Counter

# ---------------------------------------------------------------------------
# TEXT PREPROCESSING
# ---------------------------------------------------------------------------

# Transform the input text into a list of lowercase tokens (Only alphabetic sequences)
def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z]+", text.lower())

# Build the vocabulary and negative sampling distribution from the token list.
def build_vocab(tokens: list[str], min_count: int = 5):
    counts = Counter(tokens) #Count number of times that each word appears in the original text
    vocab = [w for w, c in counts.items() if c >= min_count] #Keep only words that appear at least the min_count number of times
    vocab.sort() #Sort the vocabulary alphabetically                           

    word2idx = {w: i for i, w in enumerate(vocab)} # Assign a unique index to each word in the vocabulary
    idx2word = {i: w for w, i in word2idx.items()} # Create a reverse mapping from indices back to words

    # Raise word frequencies to the 0.75 power to artificially increase the probability of sampling rare words as negatives
    freqs = np.array([counts[w] ** 0.75 for w in vocab], dtype=np.float64) 
    # Normalize the adjusted frequencies so they sum to 1, creating a valid probability distribution
    neg_sampling_probs = freqs / freqs.sum() 

    return word2idx, idx2word, neg_sampling_probs

# Generate (center, context) pairs from the token list using a random window approach.
def build_pairs(tokens: list[str], word2idx: dict, window: int = 5):

    indexed = [word2idx[t] for t in tokens if t in word2idx] 
    pairs = []
    for i, center in enumerate(indexed):
        radius = np.random.randint(1, window + 1) # Randomly choose a window size for each center word
        start = max(0, i - radius) # Define the lower bound of the context window
        end = min(len(indexed), i + radius + 1) # Define the upper bound of the context window
        for j in range(start, end):
            if j != i:
                pairs.append((center, indexed[j])) # Append the (center, context) pair to the list of pairs
    return pairs

# ---------------------------------------------------------------------------
# MODEL PARAMETERS
# ---------------------------------------------------------------------------

class SkipGramNS:

    # Initialize the model with random embeddings for both center and context words.
    def __init__(self, vocab_size: int, embed_dim: int, seed: int = 42):
        rng = np.random.default_rng(seed) # Create a random number generator with the specified seed for reproducibility
        # Randomly initialize center and context weight matrices with a uniform distribution over [-0.5/embed_dim, 0.5/embed_dim]
        self.W  = rng.uniform(-0.5 / embed_dim, 0.5 / embed_dim, (vocab_size, embed_dim))   # center embeddings
        self.Wp = rng.uniform(-0.5 / embed_dim, 0.5 / embed_dim, (vocab_size, embed_dim))   # context embeddings

    # -----------------------------------------------------------------------
    # SIGMOID
    # -----------------------------------------------------------------------

    # Sigmoid function implemented in a numerically stable way to avoid overflow issues.
    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        return np.where(x >= 0,
                        1.0 / (1.0 + np.exp(-x)),
                        np.exp(x) / (1.0 + np.exp(x)))

    # -----------------------------------------------------------------------
    # FORWARD PASS + LOSS + GRADIENTS
    # -----------------------------------------------------------------------

    def forward_backward(self,
                         center_idx: int,
                         context_idx: int,
                         neg_indices: np.ndarray):
        v_c   = self.W[center_idx] # Vector for the center word
        v_o   = self.Wp[context_idx] # Vector for the positive context word
        v_neg = self.Wp[neg_indices] # Matrix of vectors for the K negative samples 

        # --- Forward ---
        s_pos  = v_o @ v_c # Scalar score for the positive pair: v_o · v_c
        s_neg  = v_neg @ v_c # Scalar scores for the negative pairs: v_k · v_c for each k (k is the index of the negative sample)
        
        sig_pos  = self.sigmoid(s_pos)      # Calculated sigmoid for the positive pair: σ(v_o · v_c)
        sig_negs = self.sigmoid(s_neg)      # Calculated sigmoid for the negative pairs: σ(v_k · v_c) for each k

        # Loss = -log σ(s_pos) - Σ log σ(-s_neg)
        # Binary cross-entropy with "labels" [1 for pos, 0 for negs]
        loss = -np.log(sig_pos + 1e-10) - np.sum(np.log(1.0 - sig_negs + 1e-10))

        # --- Backward ---
        # ∂L/∂v_o  = -(1 - σ(s_pos)) · v_c
        grad_vo    = -(1.0 - sig_pos) * v_c # Compute the gradient with respect to the positive context embedding

        # ∂L/∂v_k  =  σ(s_neg_k) · v_c   for each negative k
        grad_vnegs = sig_negs[:, None] * v_c[None, :] # Compute the gradients with respect to each negative context embedding

        # ∂L/∂v_c  = -(1 - σ(s_pos))·v_o  +  Σ_k σ(s_neg_k)·v_k
        grad_vc    = (-(1.0 - sig_pos) * v_o + np.sum(sig_negs[:, None] * v_neg, axis=0)) # Compute the gradient with respect to the center embedding

        return loss, grad_vc, grad_vo, grad_vnegs

    # -----------------------------------------------------------------------
    # PARAMETER UPDATE (SGD)
    # -----------------------------------------------------------------------

    # Apply the computed gradients to update the embeddings using SGD.
    def update(self,
               center_idx: int,
               context_idx: int,
               neg_indices: np.ndarray,
               lr: float):
        
        # Perform the forward and backward pass to compute the loss and gradients for the given center, context, and negative samples
        loss, grad_vc, grad_vo, grad_vnegs = self.forward_backward(center_idx, context_idx, neg_indices)

        self.W[center_idx]   -= lr * grad_vc # Update the center word embedding
        self.Wp[context_idx] -= lr * grad_vo # Update the positive context word embedding

        np.add.at(self.Wp, neg_indices, -lr * grad_vnegs) # Update the negative context word embeddings (accumulates per occurrence, duplicates in neg_indices are added multiple times)

        return loss

# ---------------------------------------------------------------------------
# TRAINING LOOP
# ---------------------------------------------------------------------------

# The main training loop
def train(corpus_path: str,
          embed_dim:   int   = 100,
          window:      int   = 5,
          num_neg:     int   = 5,
          lr:          float = 0.025,
          min_lr:      float = 0.0001,
          epochs:      int   = 5,
          min_count:   int   = 5,
          seed:        int   = 42):

    np.random.seed(seed) # Set the random seed for reproducibility of results across runs.

    # -- Data --
    print("Loading and tokenizing corpus …")
    with open(corpus_path, encoding="utf-8") as f:
        raw = f.read()
    tokens = tokenize(raw) # Call tokenize function to convert the raw text into a list of tokens
    print(f"  {len(tokens):,} tokens")

    word2idx, idx2word, neg_probs = build_vocab(tokens, min_count) # Build the vocabulary and negative sampling distribution from the token list
    V = len(word2idx) # Get the size of the vocabulary
    print(f"  Vocabulary size: {V:,}")

    print("Building training pairs …")
    pairs = build_pairs(tokens, word2idx, window) # Generate (center, context) pairs from the token list using a random window approach
    print(f"  {len(pairs):,} (center, context) pairs")

    # -- Model --
    model = SkipGramNS(vocab_size=V, embed_dim=embed_dim, seed=seed) # Initialize the SkipGramNS model with the vocabulary size and embedding dimension

    total_steps  = epochs * len(pairs) # Calculate the total number of training steps
    step         = 0 
    log_interval = max(1, len(pairs) // 10) # Set the interval for printing the training progress (every 10% of the total pairs)

    print("\nTraining …")
    
    # Loop over epochs
    for epoch in range(1, epochs + 1):
        np.random.shuffle(pairs) # Shuffle the training pairs at the beginning of each epoch        
        epoch_loss = 0.0
        t0 = time.time()

        for center_idx, context_idx in pairs:
            # Linear LR decay
            progress = step / total_steps # Calculate the progress of training as a fraction of total steps completed
            current_lr = max(min_lr, lr * (1.0 - progress)) # Decay the learning rate linearly from `lr` to `min_lr` based on the training progress

            # Sample K negatives from the unigram^(3/4) distribution,
            # re-drawing if we accidentally sample the positive word
            
            # K negative word IDs sampled from neg_probs
            neg_indices = np.random.choice(V, size=num_neg, p=neg_probs, replace=True) 
            # Shift accidental positive samples to the next vocabulary index to prevent false penalization
            neg_indices[neg_indices == context_idx] = ((neg_indices[neg_indices == context_idx] + 1) % V)

            loss = model.update(center_idx, context_idx, neg_indices, current_lr) # Update the model parameters 
            epoch_loss += loss # Accumulate the loss for the current epoch
            step += 1

            # Print training progress at regular intervals
            if step % log_interval == 0:
                avg = epoch_loss / (step - (epoch - 1) * len(pairs))
                print(f"  epoch {epoch} | step {step:>8,} | "
                      f"avg loss {avg:.4f} | lr {current_lr:.5f}")

        elapsed = time.time() - t0
        print(f"Epoch {epoch} done in {elapsed:.1f}s — "
              f"avg loss {epoch_loss / len(pairs):.4f}\n")

    return model, word2idx, idx2word

# ---------------------------------------------------------------------------
# EVALUATION
# ---------------------------------------------------------------------------

# Compute cosine similarity between two vectors.
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

# Given a word, find the most similar words in the vocabulary based on cosine similarity of their embeddings.
def most_similar(word: str,
                 model: SkipGramNS,
                 word2idx: dict,
                 idx2word: dict,
                 topn: int = 10):

    # Check if the input word is in the vocabulary
    if word not in word2idx:
        raise KeyError(f"'{word}' not in vocabulary")
    idx  = word2idx[word]
    vec  = model.W[idx]
    
    # Vectorised cosine similarity against all words
    norms = np.linalg.norm(model.W, axis=1, keepdims=True) + 1e-10 # Compute the norms of all word embeddings in the vocabulary  
    sims  = model.W @ vec / (norms.squeeze() * np.linalg.norm(vec) + 1e-10) # Compute cosine similarity between the input word's embedding and all other word embeddings in the vocabulary
    sims[idx] = -1.0 # exclude the word itself
    top_indices = np.argsort(sims)[::-1][:topn] # Get the indices of the top `topn` most similar words based on cosine similarity
    return [(idx2word[i], float(sims[i])) for i in top_indices]

# Solve an analogy task of the form "a is to b as c is to ?" using vector arithmetic on the embeddings.
def analogy(a: str, b: str, c: str,
            model: SkipGramNS,
            word2idx: dict,
            idx2word: dict,
            topn: int = 5):

    # Check if all input words are in the vocabulary
    for w in (a, b, c):
        if w not in word2idx:
            raise KeyError(f"'{w}' not in vocabulary")
    
    query = (model.W[word2idx[b]] - model.W[word2idx[a]] + model.W[word2idx[c]]) # Compute the query vector using vector arithmetic: v_b - v_a + v_c
    norms = np.linalg.norm(model.W, axis=1) + 1e-10 # Compute the norms of all word embeddings in the vocabulary
    sims  = model.W @ query / (norms * np.linalg.norm(query) + 1e-10) # Compute cosine similarity between the query vector and all word embeddings in the vocabulary
    
    # Exclude input words
    for w in (a, b, c):
        sims[word2idx[w]] = -1.0
    top_indices = np.argsort(sims)[::-1][:topn] 
    
    return [(idx2word[i], float(sims[i])) for i in top_indices]

# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    model, word2idx, idx2word = train(
        corpus_path = "synthetic.txt",
        embed_dim   = 10,      
        window      = 3,
        num_neg     = 5,
        lr          = 0.025,
        min_lr      = 0.001,
        epochs      = 35,     
        min_count   = 1,       
    )

    # Example Evaluations
    print("\n=== Most similar to 'king' ===")
    
    # Print the most similar words to "king" based on cosine similarity of their embeddings
    for word, sim in most_similar("king", model, word2idx, idx2word, topn=5):
        print(f"  {word:<15} {sim:.4f}")

    # Print the result of the analogy task "king is to man as woman is to ?"
    print("\n=== Analogy: king - man + woman = ? ===")
    for word, sim in analogy("man", "king", "woman", model, word2idx, idx2word):
        print(f"  {word:<15} {sim:.4f}")