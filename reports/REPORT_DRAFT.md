# U2T02: SimCSE Sentence Embedding Model Report

**Team Members:**
- Russel Ku (Russ)
- Joni
- Rivaldo
- Damian
- Bianca

---

## Part 1: Understand the Objective

### 1. Contrastive Loss Components (Eq. 1 & Eq. 5)
- **Numerator ($\exp(\text{sim}(h_i, h_i^+) / \tau)$):** Measures the exponentiated cosine similarity between the anchor sentence representation $h_i$ and its positive counterpart $h_i^+$ (produced via an independent dropout mask in unsupervised SimCSE, or an entailment hypothesis in supervised SimCSE), scaled by temperature $\tau$.
- **Denominator ($\sum_{j=1}^N \exp(\text{sim}(h_i, h_j^+) / \tau)$):** Sums the exponentiated similarities between anchor $h_i$ and all $N$ representations in the mini-batch (including the 1 positive pair and $N-1$ in-batch negative pairs).
- **Number of Negatives Pushed Away:**
  - In **Unsupervised SimCSE** with mini-batch size $N = 64$: each anchor is pushed away from **$N - 1 = 63$ in-batch negatives**.
  - In **Supervised SimCSE with Hard Negatives** (batch size $N = 64$ pairs): each anchor has $(N - 1) + K$ negatives, where $K$ is the number of real contradictions available in that batch. This equals 127 only when all 64 pairs have contradictions. In our subset, only 28.45% of pairs have one; missing contradictions are not replaced by positive hypotheses.

### 2. Temperature Parameter ($\tau$)
- **Function:** Temperature $\tau$ controls the "softness" vs. "hardness" of the probability distribution over negatives. Smaller $\tau$ sharpens the softmax, heavily penalizing negatives that are deceptively close in cosine distance (hard negatives).
- **Paper Findings:** Gao et al. (2021) ablated $\tau \in [0.01, 0.5]$ and found that $\tau = 0.05$ yields optimal performance. A $\tau$ that is too large (e.g., $0.1 - 0.5$) smooths out gradients, failing to separate hard negatives; a $\tau$ that is too small ($< 0.01$) leads to unstable, saturated gradients.

### 3. SimCSE Contrastive Objective vs. Sentence-BERT (2019)
- **SBERT (2019):** Trains with a 3-way classification loss over concatenated vector features $(u, v, |u - v|)$ passed through a linear classification layer to predict entailment, neutral, or contradiction.
- **SimCSE (2021):** Employs an InfoNCE contrastive objective that directly maximizes the cosine similarity between normalized positive pairs and minimizes similarity against negatives on the unit hypersphere.
- **Impact on STS-B:** SBERT's classification head allows the encoder to embed sentences in an anisotropic space (narrow cone), relying on the linear classifier to separate classes. In contrast, STS-B directly tests raw cosine similarity between embeddings without any classification head. SimCSE explicitly regularizes the representation space to be isotropic and uniform, yielding superior rank correlation on STS-B.

### 4. Relation to Alignment and Uniformity (Wang & Isola, 2020)
- **Alignment:** Asymptotically achieved by the numerator, minimizing the distance between positive representations $\mathbb{E}_{(x, x^+)}[\|f(x) - f(x^+)\|^2]$.
- **Uniformity:** Asymptotically achieved by the denominator, pushing apart representations of all distinct instances to distribute embeddings uniformly across the unit sphere $\log \mathbb{E}_{x, y}[\exp(-2\|f(x) - f(y)\|^2)]$, avoiding representation collapse.

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
| **Raw `bert-base-uncased` (mean pooling)** | 59.31 | 47.29 | 0.3678 / 0.3044 | -1.6348 / -1.6186 |
| **SBERT-2019 (`bert-base-nli-mean-tokens`)** | 80.77 | 76.98 | 0.6996 / 0.5498 | -3.0557 / -3.0493 |
| **Our Unsupervised SimCSE** | **76.32** | **67.32** | 0.6829 / 0.6090 | -2.8887 / -2.8900 |
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
  - **Alignment ($\alpha=2$):** **0.6090**
  - **Uniformity ($t=2$):** **-2.8900**
  - **Live Inference Verification:** Passed semantic similarity demo ("Dog running" vs "Puppy playing" cosine similarity = $0.5833$).
