# U2T02: SimCSE Sentence Embedding Model Report (working draft)

> **Superseded by [FINAL_REPORT.md](FINAL_REPORT.md)**, which is the version to submit. Part 1 below is the final reviewed text; the alignment values here were corrected (STS-B score-scale bug, see `reports/QA_CHECKLIST.md`).

**Team Members:**
- Russel Ku (Russ)
- Joni
- Rivaldo
- Damian
- Bianca

---

## Part 1: Understand the Objective (final version, reviewed by Bianca)

### 1.1 Eq. 1 and Eq. 5: numerator, denominator and number of negatives

Unsupervised SimCSE (Gao et al., 2021, Eq. 1) minimizes, for each sentence $x_i$ in a mini-batch of $N$ sentences,

$$\ell_i = -\log \frac{e^{\mathrm{sim}(h_i,\,h_i^{+})/\tau}}{\sum_{j=1}^{N} e^{\mathrm{sim}(h_i,\,h_j^{+})/\tau}}, \qquad \mathrm{sim}(a,b)=\frac{a^\top b}{\lVert a\rVert\,\lVert b\rVert}.$$

- **Numerator:** the exponentiated, temperature-scaled cosine similarity between the anchor $h_i$ and its own positive $h_i^{+}$. In the unsupervised case, $h_i$ and $h_i^{+}$ are two encodings of the same sentence with **different dropout masks** $z, z'$. Dropout is the only data augmentation.
- **Denominator:** a sum over all $N$ positives of the batch, $h_1^{+},\dots,h_N^{+}$. It includes the anchor's own positive ($j=i$) and the $N-1$ positives of the other sentences, which serve as **in-batch negatives**. The loss is the cross-entropy of a softmax classifier that must pick $h_i^{+}$ among $N$ candidates.
- **Negatives per step with our batch size ($N=64$):** each anchor is pushed away from **$N-1 = 63$ negatives**. One optimization step has $64 \times 63 = 4{,}032$ anchor–negative pairs.

The supervised version (Eq. 5) uses NLI triplets: the premise is the anchor, its entailment hypothesis is the positive $h_i^{+}$, and its contradiction hypothesis is a hard negative $h_i^{-}$:

$$\ell_i = -\log \frac{e^{\mathrm{sim}(h_i,h_i^{+})/\tau}}{\sum_{j=1}^{N}\left(e^{\mathrm{sim}(h_i,h_j^{+})/\tau}+e^{\mathrm{sim}(h_i,h_j^{-})/\tau}\right)}.$$

With full triplets the denominator has $2N$ terms, so each anchor has $(N-1)+N = 127$ negatives at $N=64$. In our SNLI subset only **9,488 of 33,351 pairs (28.45 %)** have a contradiction. We do not invent missing negatives and never substitute a positive for one. Each anchor therefore sees $63 + K$ negatives, where $K$ is the number of real contradictions in that batch. With uniform shuffling $K \sim \mathrm{Binomial}(64, 0.2845)$, so $\mathbb{E}[K] \approx 18.2$ (sd ≈ 3.6). That gives **≈ 81 negatives per anchor on average**, instead of the paper's 127.

### 1.2 The temperature τ

$\tau$ scales the cosine logits before the softmax. Cosine lies in $[-1,1]$, so the logits span a range of $2/\tau$: 40 at $\tau=0.05$, but only 2 at $\tau=1$. The gradient on negative $j$ is weighted by its softmax probability $p_{ij}\propto e^{\mathrm{sim}(h_i,h_j)/\tau}$:

- **Small τ** sharpens the softmax. Almost all gradient goes to the hardest negatives (the most similar ones), so the loss acts like a hard-negative-mining loss. If τ is too small, training focuses on a few, possibly false, negatives and becomes noisy.
- **Large τ** flattens the softmax. All negatives are pushed with similar weight, and the model cannot make the positive's probability approach 1 (at τ = 1 the largest possible logit ratio is $e^{2}\approx 7.4$), so the space is only weakly reshaped.

**What the paper found** (Appendix D, Table D.1, supervised SimCSE-BERT$_\text{base}$, STS-B dev): τ = 0.001 → 84.9, 0.01 → 85.4, **0.05 → 86.2**, 0.1 → 82.0, 1 → 64.0. Dot product without normalization or temperature ("N/A") scored 85.9. Cosine similarity with a well-tuned **τ = 0.05** is best, and a large τ is catastrophic.

**Our data agrees.** In the supervised sweep (dev, seed 42), τ = 0.05 was best at both learning rates: 81.45 / **81.67** / 80.49 for τ = 0.03 / 0.05 / 0.10 at lr 3e-5, and 81.04 / **81.23** / 80.27 at lr 5e-5. Going from 0.05 to 0.10 costs more (≈ 1.0–1.2 points) than going from 0.05 to 0.03 (≈ 0.2), the same asymmetry the paper reports.

### 1.3 SimCSE vs. the Sentence-BERT (2019) objective, and why it matters for STS-B

- **SBERT** (Reimers & Gurevych, 2019) mean-pools BERT into $u$ and $v$. On NLI it trains a **3-way softmax classifier** $o=\mathrm{softmax}(W_t[u;v;|u-v|])$ with cross-entropy over {entailment, neutral, contradiction}. Each training example is a single pair with a single label. The classifier $W_t$ is **thrown away** at inference, and sentences are then compared with cosine similarity.
- **SimCSE** trains with InfoNCE directly on the **cosine similarity of the normalized embeddings**. Every step contrasts each anchor with all other sentences in the batch.

**Why it matters for STS-B.** STS-B is evaluated with no regressor: embed, normalize, take the cosine, compute Spearman (Gao et al., Appendix B; we use the same protocol). The score therefore depends only on the geometry of the embedding space.

1. SimCSE optimizes **exactly the test-time function** (cosine). SBERT optimizes a linear classifier on top of $[u;v;|u-v|]$, and nothing in its loss forces $\cos(u,v)$ to be monotonic in semantic similarity; good cosine behavior is only a by-product.
2. A classification loss only needs the three classes to be linearly separable for each pair. It does not have to spread *unrelated* sentences across the sphere. SimCSE's in-batch negatives explicitly penalize every unrelated pair that sits too close, which is what reduces BERT's anisotropy (see 1.4).

Our numbers reflect this. Our supervised SimCSE, trained on 33,351 SNLI pairs, reaches **79.02 test** vs. **76.98** for SBERT-2019, which was trained on all of SNLI + MultiNLI (~1M labeled pairs). Its uniformity is better on dev (−3.137 vs. −3.056) and equal within 0.003 on test (−3.046 vs. −3.049).

### 1.4 Relation to alignment and uniformity (Wang & Isola, 2020)

For L2-normalized encoders $f$:

$$\ell_\text{align}=\mathbb{E}_{(x,x^+)\sim p_\text{pos}}\lVert f(x)-f(x^+)\rVert^2, \qquad \ell_\text{uniform}=\log\,\mathbb{E}_{x,y\sim p_\text{data}}\,e^{-2\lVert f(x)-f(y)\rVert^2}.$$

Lower is better for both. **Alignment** measures how close positives are, and **uniformity** measures how evenly embeddings cover the hypersphere. As the number of negatives grows, the contrastive objective becomes (SimCSE Eq. 6):

$$-\frac{1}{\tau}\,\mathbb{E}_{(x,x^+)}\!\left[f(x)^\top f(x^+)\right] + \mathbb{E}_{x}\!\left[\log \mathbb{E}_{x^-}\, e^{f(x)^\top f(x^-)/\tau}\right].$$

- **First term (numerator) = alignment.** For unit vectors $\lVert f(x)-f(x^+)\rVert^2 = 2-2f(x)^\top f(x^+)$, so this term equals $\frac{1}{\tau}\left(\tfrac{1}{2}\ell_\text{align}-1\right)$. Minimizing the loss minimizes alignment exactly.
- **Second term (denominator) ≈ uniformity.** With $t=\tfrac{1}{2\tau}$, $\ell_\text{uniform} = \log \mathbb{E}_{x,y} e^{f(x)^\top f(y)/\tau} - \tfrac{1}{\tau}$. By Jensen's inequality the second term is upper-bounded by $\ell_\text{uniform}+\tfrac{1}{\tau}$. Pushing negatives apart therefore drives the representation toward uniformity and prevents collapse.
- **Anisotropy.** Gao et al. also show this term upper-bounds the sum of all entries of $WW^\top$ (the embedding similarity matrix). Minimizing it flattens the singular-value spectrum, which is a direct remedy for BERT's "narrow cone".

**What we measured.** Raw BERT has uniformity −1.62 on test, and nearly all its pair cosines are above 0.5: the anisotropy problem. Every contrastive model reaches −2.89 to −3.14. Alignment alone is misleading: raw BERT has a *good-looking* alignment (0.22) only because everything is close to everything. The two metrics must be read together, as in the plot in Part 6.

---

## Part 2 & 3: Training Configurations and Run Logs

| Run ID | Mode | Backbone | Batch Size | LR | Epochs | $\tau$ | Pooling | Dev Spearman | Test Spearman |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `unsup_simcse_default` | Unsup | `bert-base-uncased` | 64 | 3e-5 | 1 | 0.05 | CLS | **76.32** | **67.32** |
| `jonav_sup_hnon_lr3e-05_tau0.05_bs64_ep3_seed42` | Sup (Hard Negs) | `bert-base-uncased` | 64 | 3e-5 | 3 | 0.05 | CLS + MLP | **81.67** | **79.02** |
| `ablation_unsup_same_mask` | Unsup (Ablation) | `bert-base-uncased` | 64 | 3e-5 | 1 | 0.05 | CLS | **55.39** | **47.78** |
| `jonav_sup_hnoff_lr3e-05_tau0.05_bs64_ep3_seed42` | Sup (Ablation) | `bert-base-uncased` | 64 | 3e-5 | 3 | 0.05 | CLS + MLP | **81.01** | **77.11** |

The supervised learning rate and temperature were chosen from six runs using only STS-B dev. The two final seed-42 checkpoints were locked before test evaluation. Additional seeds estimate dev variability and do not replace the selected final pair. See [Jonav's report and complete sweep](JONAV_RESULTADOS.md).

---

## Part 4 & 5: Ablation Studies

### Ablation 1 (Unsupervised): Independent Dropout vs. Same Dropout Mask
- **Hypothesis:** Without independent dropout noise, both views are identical ($z_1 = z_2$). The contrastive loss loses its data augmentation and representation learning collapses towards trivial shortcuts.
- **Delta Observed:** $\Delta \text{Dev} = \mathbf{+20.93}$ points ($76.32 \to 55.39$), $\Delta \text{Test} = \mathbf{+19.54}$ points ($67.32 \to 47.78$).
- **Theoretical Insight:** With `same_dropout_mask=True`, the test performance drops to $47.78$, which is virtually indistinguishable from untrained raw BERT ($47.29$). This empirically proves Gao et al.'s premise: standard dropout acts as minimal data augmentation; without representation perturbation between positive views, the InfoNCE numerator $\text{sim}(h_i, h_i^+)$ trivially equals $1.0$, preventing meaningful gradient flow.

### Ablation 2 (Supervised): Hard Negatives ON vs. OFF
- **Hypothesis:** Contradiction pairs force the model to distinguish fine-grained semantic opposites that share substantial lexical overlap (e.g., "A dog running" vs. "A dog sleeping").
- **Delta Observed (ON − OFF, seed 42):** $\Delta \text{Dev} = +0.66$, $\Delta \text{Test} = +1.91$ points.
- **Dev variability:** Paired deltas for seeds 42, 123, 456 are +0.66, +0.83, +0.11. Their mean is +0.53 and sample standard deviation is 0.38. ON/OFF dev sample deviations are 0.25/0.54. All three deltas are positive, but three seeds provide descriptive evidence rather than statistical significance. There are no extra test evaluations for seeds 123 and 456.
- **Geometry:** ON improves uniformity on dev and test. Alignment is slightly worse on dev (0.1857 vs. 0.1799) and slightly better on test (0.1903 vs. 0.1914); the gain does not improve every metric consistently. Contradictions plausibly help distinguish semantically incompatible sentences, but retrieval cases are still needed to assess this explanation.

---

## Part 6: Benchmark Table and The Gap Analysis

### Benchmark Comparison Table
| Model | Dev Spearman ($\times 100$) | Test Spearman ($\times 100$) | Dev / Test Alignment ($\alpha=2$) | Dev / Test Uniformity ($t=2$) |
| :--- | :--- | :--- | :--- | :--- |
| **Raw `bert-base-uncased` (mean pooling)** | 59.31 | 47.29 | 0.1948 / 0.2155 | -1.6348 / -1.6186 |
| **SBERT-2019 (`bert-base-nli-mean-tokens`)** | 80.77 | 76.98 | 0.1929 / 0.1798 | -3.0557 / -3.0493 |
| **Our Unsupervised SimCSE** | **76.32** | **67.32** | 0.3150 / 0.3501 | -2.8887 / -2.8900 |
| **Our Supervised SimCSE** | **81.67** | **79.02** | 0.1857 / 0.1903 | -3.1374 / -3.0460 |
| **SimCSE Paper Unsupervised (Gao et al. 2021)** | 82.50 | 76.85 | - | - |
| **SimCSE Paper Supervised BERT-base (Gao et al. 2021)** | 86.20 | 84.25 | - | - |

Paper references use STS-B specifically: supervised dev 86.2 is from Table 4 and test 84.25 from Table 5. The value 81.57 is the average of seven STS tasks, not STS-B test. Our supervised gaps are **4.53 dev / 5.23 test** points. See the [original paper](https://aclanthology.org/2021.emnlp-main.552.pdf).

### Accounting for the Gap (Critical Analysis)
*These protocol differences may contribute to the gap; our experiments do not isolate their effects:*

1. **Training data:** The paper uses one million Wikipedia sentences for unsupervised training and approximately 314k SNLI+MNLI positive pairs for supervised training. Our SNLI subset yields 165,529 unique sentences and 33,351 entailment pairs.
2. **Batch and tuning:** Appendix A uses batch 64 for unsupervised BERT-base and 512 for supervised BERT-base. Our batch is 64 for both. The paper notes that batch sensitivity depends on learning-rate tuning, so a smaller batch alone does not establish the cause of the gap.
3. **Hard negatives:** Only 9,488 of our 33,351 entailment pairs (28.45%) have a matched contradiction. This changes the coverage and number of available hard negatives relative to the full NLI setup.

Our supervised model exceeds the team's recorded SBERT-2019 test baseline by **2.04 points** and raw BERT by **31.73 points**. These are comparisons with existing baseline logs, not newly rerun baselines. Supervised interpretation and reproducibility evidence are documented in [JONAV_RESULTADOS.md](JONAV_RESULTADOS.md).

---

## Part 7: Hugging Face Publication & Verification

- **Hub Model Repository:** [RusselKuAguilar/simcse-bert-uncased-unsup](https://huggingface.co/RusselKuAguilar/simcse-bert-uncased-unsup)
- **Verification Result:** Reloaded model directly from the Hugging Face Hub via `SentenceTransformer("RusselKuAguilar/simcse-bert-uncased-unsup")` and re-evaluated on STS-B test split.
  - **Reloaded Test Spearman:** **67.32** (Exact match against local checkpoint $67.32$, $\Delta = 0.00$).
  - **Alignment ($\alpha=2$):** **0.3501** (pairs with score ≥ 4; the 0.6090 in the training log used the old all-pairs definition)
  - **Uniformity ($t=2$):** **-2.8900**
  - **Live Inference Verification:** Passed semantic similarity demo ("Dog running" vs "Puppy playing" cosine similarity = $0.5833$).
