"""
model.py — N-gram Language Model for Next Word Prediction
Built from scratch using only Python stdlib + numpy.

Concepts used:
  - Tokenisation (regex-based)
  - Unigram, Bigram, Trigram frequency counts
  - Laplace (add-k) smoothing
  - Perplexity for model evaluation
  - Backoff: trigram → bigram → unigram
"""

import re
import json
import math
import random
import numpy as np
from collections import defaultdict, Counter


# ── Tokeniser ────────────────────────────────────────────────────────────────

def tokenise(text: str) -> list[str]:
    """
    Lowercase, keep apostrophes, strip everything else non-alpha.
    Returns a flat list of word tokens.
    """
    text = text.lower()
    text = re.sub(r"[^a-z' ]", " ", text)
    tokens = text.split()
    # Remove standalone apostrophes
    tokens = [t.strip("'") for t in tokens if t.strip("'")]
    return tokens


def sentence_tokenise(text: str) -> list[list[str]]:
    """
    Split text into sentences, then tokenise each.
    Adds <s> (start) and </s> (end) boundary tokens.
    """
    # Split on '.', '!', '?', ';'
    sentences = re.split(r'[.!?;]+', text)
    result = []
    for sent in sentences:
        tokens = tokenise(sent)
        if tokens:
            result.append(["<s>"] + tokens + ["</s>"])
    return result


# ── N-gram Model ─────────────────────────────────────────────────────────────

class NGramModel:
    """
    Trigram language model with Laplace smoothing and stupid backoff.

    Attributes:
        n          : maximum n-gram order (default 3)
        k          : Laplace smoothing constant (default 0.1)
        unigrams   : Counter of single tokens
        bigrams    : defaultdict(Counter) — bigrams[(w1)][w2] = count
        trigrams   : defaultdict(Counter) — trigrams[(w1,w2)][w3] = count
        vocab      : set of all known words
        vocab_size : |vocab|
    """

    def __init__(self, n: int = 3, k: float = 0.1):
        self.n = n
        self.k = k
        self.unigrams: Counter = Counter()
        self.bigrams:  defaultdict = defaultdict(Counter)
        self.trigrams: defaultdict = defaultdict(Counter)
        self.vocab:    set = set()
        self.vocab_size: int = 0
        self._trained  = False

    # ── Training ──────────────────────────────────────────────────────────────

    def train(self, text: str) -> dict:
        """
        Train on raw text. Returns training statistics.
        """
        sentences = sentence_tokenise(text)
        total_tokens = 0

        for sent in sentences:
            tokens = sent
            total_tokens += len(tokens)

            # Unigrams
            for w in tokens:
                self.unigrams[w] += 1
                self.vocab.add(w)

            # Bigrams: (w1) → w2
            for i in range(len(tokens) - 1):
                self.bigrams[tokens[i]][tokens[i+1]] += 1

            # Trigrams: (w1, w2) → w3
            for i in range(len(tokens) - 2):
                ctx = (tokens[i], tokens[i+1])
                self.trigrams[ctx][tokens[i+2]] += 1

        # Remove boundary tokens from vocab for prediction
        self.vocab.discard("<s>")
        self.vocab.discard("</s>")
        self.vocab_size = len(self.vocab)
        self._trained = True

        return {
            "sentences":   len(sentences),
            "total_tokens": total_tokens,
            "vocab_size":  self.vocab_size,
            "unique_bigrams":  sum(len(v) for v in self.bigrams.values()),
            "unique_trigrams": sum(len(v) for v in self.trigrams.values()),
        }

    # ── Probability ───────────────────────────────────────────────────────────

    def _p_unigram(self, word: str) -> float:
        """P(word) with Laplace smoothing."""
        total = sum(self.unigrams.values())
        return (self.unigrams[word] + self.k) / (total + self.k * self.vocab_size)

    def _p_bigram(self, prev: str, word: str) -> float:
        """P(word | prev) with Laplace smoothing."""
        ctx_count = sum(self.bigrams[prev].values())
        return (self.bigrams[prev][word] + self.k) / (ctx_count + self.k * self.vocab_size)

    def _p_trigram(self, prev2: str, prev1: str, word: str) -> float:
        """P(word | prev2, prev1) with Laplace smoothing."""
        ctx = (prev2, prev1)
        ctx_count = sum(self.trigrams[ctx].values())
        return (self.trigrams[ctx][word] + self.k) / (ctx_count + self.k * self.vocab_size)

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict(self, context: str, top_k: int = 5) -> list[dict]:
        """
        Given a context string, predict the top_k most likely next words.
        Uses trigram → bigram → unigram backoff with stupid-backoff weights.

        Returns list of {word, score, source} dicts sorted by score desc.
        """
        if not self._trained:
            return []

        tokens = tokenise(context)
        if not tokens:
            # No context: return most common words
            return [
                {"word": w, "score": round(c / sum(self.unigrams.values()), 4), "source": "unigram"}
                for w, c in self.unigrams.most_common(top_k)
                if w not in ("<s>", "</s>")
            ]

        scores: dict[str, float] = {}

        # Trigram context
        if len(tokens) >= 2:
            w2, w1 = tokens[-2], tokens[-1]
            ctx = (w2, w1)
            if ctx in self.trigrams and sum(self.trigrams[ctx].values()) > 0:
                for word, count in self.trigrams[ctx].items():
                    if word not in ("<s>", "</s>"):
                        scores[word] = scores.get(word, 0) + 0.6 * self._p_trigram(w2, w1, word)

        # Bigram context
        if tokens:
            w1 = tokens[-1]
            if w1 in self.bigrams and sum(self.bigrams[w1].values()) > 0:
                for word, count in self.bigrams[w1].items():
                    if word not in ("<s>", "</s>"):
                        scores[word] = scores.get(word, 0) + 0.3 * self._p_bigram(w1, word)

        # Unigram fallback (always contributes a little)
        for word, count in self.unigrams.most_common(50):
            if word not in ("<s>", "</s>"):
                scores[word] = scores.get(word, 0) + 0.1 * self._p_unigram(word)

        # Sort and label source
        sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        results = []
        for word, score in sorted_words[:top_k]:
            # Determine primary source
            if len(tokens) >= 2 and (tokens[-2], tokens[-1]) in self.trigrams and word in self.trigrams[(tokens[-2], tokens[-1])]:
                source = "trigram"
            elif tokens and tokens[-1] in self.bigrams and word in self.bigrams[tokens[-1]]:
                source = "bigram"
            else:
                source = "unigram"
            results.append({"word": word, "score": round(score, 6), "source": source})

        return results

    def predict_sequence(self, seed: str, length: int = 10, temperature: float = 1.0) -> str:
        """
        Auto-complete: generate `length` words after `seed`.
        temperature > 1 = more random, < 1 = more deterministic.
        """
        tokens = tokenise(seed)
        generated = tokens[:]

        for _ in range(length):
            context = " ".join(generated)
            candidates = self.predict(context, top_k=10)
            if not candidates:
                break

            # Temperature sampling
            words  = [c["word"]  for c in candidates]
            logits = np.array([c["score"] for c in candidates], dtype=float)
            logits = np.log(logits + 1e-10) / temperature
            logits -= logits.max()
            probs  = np.exp(logits)
            probs /= probs.sum()

            chosen = np.random.choice(words, p=probs)
            generated.append(chosen)

        return " ".join(generated[len(tokens):])

    # ── Evaluation ────────────────────────────────────────────────────────────

    def perplexity(self, text: str) -> float:
        """
        Compute perplexity of the model on a held-out text.
        Lower perplexity = better model fit.
        PP = exp(-1/N * sum(log P(w_i | context)))
        """
        sentences = sentence_tokenise(text)
        log_prob_sum = 0.0
        N = 0

        for sent in sentences:
            for i in range(1, len(sent)):
                word = sent[i]
                if word in ("<s>", "</s>"):
                    continue
                if i >= 2 and self.n >= 3:
                    p = self._p_trigram(sent[i-2], sent[i-1], word)
                elif i >= 1:
                    p = self._p_bigram(sent[i-1], word)
                else:
                    p = self._p_unigram(word)
                log_prob_sum += math.log(p + 1e-10)
                N += 1

        if N == 0:
            return float("inf")
        return round(math.exp(-log_prob_sum / N), 2)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str):
        """Serialise model to JSON."""
        data = {
            "n": self.n,
            "k": self.k,
            "unigrams": dict(self.unigrams),
            "bigrams":  {k: dict(v) for k, v in self.bigrams.items()},
            "trigrams": {json.dumps(list(k)): dict(v) for k, v in self.trigrams.items()},
            "vocab":    list(self.vocab),
        }
        with open(path, "w") as f:
            json.dump(data, f)

    def load(self, path: str):
        """Load model from JSON."""
        with open(path) as f:
            data = json.load(f)
        self.n = data["n"]
        self.k = data["k"]
        self.unigrams = Counter(data["unigrams"])
        self.bigrams  = defaultdict(Counter, {k: Counter(v) for k, v in data["bigrams"].items()})
        self.trigrams = defaultdict(Counter, {
            tuple(json.loads(k)): Counter(v) for k, v in data["trigrams"].items()
        })
        self.vocab      = set(data["vocab"])
        self.vocab_size = len(self.vocab)
        self._trained   = True


# ── Built-in training corpus ─────────────────────────────────────────────────

CORPUS = """
The quick brown fox jumps over the lazy dog. The dog barked loudly at the fox.
The fox ran away quickly through the dark forest. The forest was full of tall trees.
Natural language processing is a field of artificial intelligence. It helps computers
understand human language. Machine learning models can predict the next word in a sentence.
Deep learning has improved natural language processing significantly. Neural networks
learn patterns from large amounts of text data. The model predicts the probability of
each word given its context. Language models are used in autocomplete systems.
Autocomplete helps users type faster on mobile devices. The keyboard suggests the next
word based on previous words. This is called next word prediction. Prediction accuracy
improves with more training data. Training data consists of text from books and websites.
The internet contains billions of words of text. Text preprocessing removes punctuation
and converts text to lowercase. Tokenisation splits text into individual words.
N-gram models use sequences of n words to predict the next word. Bigram models use
two words of context. Trigram models use three words. Higher order models capture
longer dependencies in text. Smoothing techniques handle unseen word combinations.
Laplace smoothing adds a small count to all possible combinations. This prevents
zero probability for unseen words. Perplexity measures how well a language model
fits a given text. Lower perplexity indicates a better model. The best language models
have low perplexity on held-out test data. Recurrent neural networks improved over
traditional n-gram models. Transformers further improved language modelling.
The attention mechanism allows models to focus on relevant context. BERT and GPT
are large transformer language models. They are trained on massive text corpora.
Transfer learning allows models to be fine-tuned on specific tasks. Text generation
produces coherent sentences by sampling from the model distribution. Temperature
controls the randomness of text generation. Low temperature produces conservative
predictions. High temperature produces more creative and diverse text.
Science and technology continue to advance rapidly. Computers are becoming more
powerful every year. Artificial intelligence is transforming many industries.
Healthcare benefits from AI diagnosis tools. Education is changing with online
learning platforms. Communication has been revolutionised by the internet.
People can connect with others around the world instantly. Social media platforms
allow sharing of ideas and information. Knowledge is more accessible than ever before.
The human brain is a remarkable information processing system. Language is one of
the most complex abilities humans possess. Children learn language naturally through
exposure and practice. Reading and writing are fundamental skills for education.
Books contain the accumulated knowledge of human civilisation. Libraries preserve
important texts for future generations.
"""


def get_default_model() -> NGramModel:
    """Return a pre-trained model on the built-in corpus."""
    m = NGramModel(n=3, k=0.1)
    m.train(CORPUS)
    return m
