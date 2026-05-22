# Diffusion vs Auto-Regressive Language Models for Hindi NLP
## Comprehensive Research Document

**PI:** Prof. Debkanta Chakraborty (Georgia Institute of Technology / IIT Kanpur)
**Team:** Aditya Bhatia, Tanish Anand, Pavitra Kushwaha
**Date:** May 2026

---

## 1. Introduction & Research Motivation

### 1.1 Problem Statement

Large Language Models (LLMs) have achieved remarkable success using **auto-regressive (AR)** architectures that generate text left-to-right, one token at a time. However, recent advances in **text diffusion models** present a fundamentally different paradigm: generating all tokens simultaneously through iterative denoising.

This project systematically investigates:

> **Can text diffusion models match or outperform auto-regressive models for Hindi NLP tasks, and under what conditions does each paradigm excel?**

### 1.2 Why Hindi?

- Hindi is the 3rd most spoken language globally (600M+ speakers)
- Severely under-resourced compared to English in NLP research
- Unique challenges: Devanagari script, agglutinative morphology, word order flexibility
- Legal NLP in Hindi is virtually unexplored

### 1.3 Research Contributions

1. **First comprehensive benchmark** comparing 4 diffusion models vs 4 AR models on Hindi NLP
2. **Legal NLP in Hindi**: First application of diffusion models to Hindi legal document analysis
3. **5-step evaluation pipeline**: From zero-shot to agentic AI, revealing how models improve with adaptation
4. **Practical insights**: Which model paradigm to use for which Hindi NLP task

---

## 2. Models Under Study

### 2.1 Diffusion Language Models

#### 2.1.1 SEDD (Score Entropy Discrete Diffusion)
- **Paper:** "Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution" (ICML 2024)
- **Authors:** Aaron Lou, Chenlin Meng, Stefano Ermon (Stanford)
- **Key Innovation:** Score entropy loss for discrete spaces
- **Mechanism:** Learns ratios between token probabilities, generates by iteratively estimating discrete scores
- **Parameters:** ~350M
- **Notable:** First diffusion model to outperform GPT-2; 25-75% lower perplexity than prior diffusion methods

#### 2.1.2 LLaDA (Large Language Diffusion with mAsking)
- **Paper:** "Large Language Diffusion Models" (arXiv:2502.09992, 2025)
- **Authors:** Renmin University of China & Ant Group
- **Key Innovation:** Masking-based diffusion at LLM scale (8B parameters)
- **Mechanism:** Forward = random masking, Reverse = progressive unmasking with confidence
- **Parameters:** 8B
- **Notable:** Competitive with LLaMA 3 8B; handles bidirectional reasoning

#### 2.1.3 D3PM (Discrete Denoising Diffusion Probabilistic Models)
- **Paper:** "Structured Denoising Diffusion Models in Discrete State-Spaces" (NeurIPS 2021)
- **Authors:** Google Research
- **Key Innovation:** Structured Markov transition matrices for discrete noise
- **Mechanism:** Defines Q_t matrices (uniform/absorbing/embedding-based) for corruption and denoising
- **Parameters:** ~110M
- **Notable:** Formally connects diffusion models to masked language models (BERT)

#### 2.1.4 DiffuseLM (Diffusion-LM)
- **Paper:** "Diffusion-LM Improves Controllable Text Generation" (NeurIPS 2022)
- **Authors:** Xiang Lisa Li et al.
- **Key Innovation:** Continuous diffusion in embedding space (like image diffusion)
- **Mechanism:** Gaussian noise in embedding space → denoise → round to nearest token
- **Parameters:** ~100M
- **Notable:** Supports gradient-based controllable generation via plug-and-play classifiers

### 2.2 Auto-Regressive Models

#### 2.2.1 LLaMA 3 8B Instruct
- **Paper:** Meta AI (2024)
- **Architecture:** Causal transformer with grouped query attention
- **Parameters:** 8B
- **Context:** 8192 tokens
- **Notable:** Strong multilingual capabilities, instruction-tuned

#### 2.2.2 Gemma 2 9B Instruct
- **Paper:** Google DeepMind (2024)
- **Architecture:** Causal transformer with sliding window + global attention
- **Parameters:** 9B
- **Context:** 8192 tokens
- **Notable:** Excellent on Indic languages, knowledge distillation from Gemini

#### 2.2.3 Mistral 7B Instruct v0.3
- **Paper:** Mistral AI (2023)
- **Architecture:** Causal transformer with sliding window attention
- **Parameters:** 7B
- **Context:** 32768 tokens
- **Notable:** Efficient attention mechanism, strong reasoning

#### 2.2.4 IndicBERT v2 (AI4Bharat)
- **Paper:** "IndicNLPSuite" (2023)
- **Architecture:** Masked language model (MLM)
- **Parameters:** 278M
- **Training:** Pre-trained on 22 Indian languages including Hindi
- **Notable:** Best-in-class for Indic language understanding tasks

---

## 3. Tasks

### 3.1 Language Tasks

#### 3.1.1 English → Hindi Translation
- **Dataset:** IIT Bombay English-Hindi Parallel Corpus (1.49M sentence pairs)
- **Task Type:** Sequence-to-sequence generation
- **Metrics:** BLEU, chrF, BERTScore, COMET
- **Challenge:** Morphological richness, word order differences, honorific system

#### 3.1.2 Hindi Text Summarization
- **Dataset:** XL-Sum Hindi (from BBC Hindi), ILSUM
- **Task Type:** Abstractive summarization
- **Metrics:** ROUGE-1/2/L, BERTScore, BLEU
- **Challenge:** Preserving factual accuracy while generating fluent summaries

### 3.2 Legal Tasks

#### 3.2.1 Bail Prediction
- **Dataset:** HLDC - Hindi Legal Documents Corpus (900K+ documents from UP district courts)
- **Task Type:** Binary classification (Grant / Reject)
- **Metrics:** Accuracy, F1 (Macro), Precision, Recall, AUC-ROC
- **Challenge:** Long legal documents, domain-specific vocabulary, nuanced reasoning

#### 3.2.2 Judge Verdict Prediction
- **Dataset:** HLDC + IndianBailJudgments-1200
- **Task Type:** Multi-class classification (Convicted / Acquitted / Remanded)
- **Metrics:** Accuracy, F1 (Macro/Weighted), Confusion Matrix
- **Challenge:** Understanding complex legal arguments and precedents

---

## 4. Pipeline Steps

### 4.1 Zero-Shot (Step 1)
- Direct inference with task instruction only
- Tests inherent multilingual knowledge
- No Hindi-specific training examples

### 4.2 Few-Shot (Step 2)
- 5 carefully selected Hindi examples as demonstrations
- Tests in-context learning ability
- Same model weights, different prompting

### 4.3 Fine-Tuning (Step 3)
- **Method:** LoRA/QLoRA (4-bit quantization)
- **Hyperparameters:** lr=2e-4, epochs=3, LoRA r=16, alpha=32
- Task-specific training on Hindi datasets
- Tests adaptation capability

### 4.4 RAG (Step 4)
- **Embeddings:** paraphrase-multilingual-MiniLM-L12-v2
- **Vector Store:** FAISS
- Retrieved Hindi documents augment model context
- Tests knowledge retrieval and integration

### 4.5 Agentic AI (Step 5)
- Multi-step reasoning with chain-of-thought
- Self-reflection and answer refinement
- Tool-augmented generation

---

## 5. Evaluation Framework

### 5.1 Generation Metrics

| Metric | Description | Range | Better |
|--------|-------------|-------|--------|
| **BLEU** | N-gram precision (1-4 grams) | 0-100 | Higher |
| **ROUGE-1/2/L** | Unigram/bigram/longest common subsequence recall | 0-100 | Higher |
| **BERTScore** | Semantic similarity via contextual embeddings | 0-100 | Higher |
| **Perplexity** | Model's uncertainty on generated text | 1-∞ | Lower |
| **chrF** | Character-level F-score | 0-100 | Higher |

### 5.2 Classification Metrics

| Metric | Description | Range | Better |
|--------|-------------|-------|--------|
| **Accuracy** | Fraction of correct predictions | 0-100 | Higher |
| **F1 (Macro)** | Harmonic mean of precision and recall (unweighted) | 0-100 | Higher |
| **F1 (Weighted)** | F1 weighted by class frequency | 0-100 | Higher |
| **Precision** | Fraction of true positives among predicted positives | 0-100 | Higher |
| **Recall** | Fraction of true positives among actual positives | 0-100 | Higher |
| **AUC-ROC** | Area under the receiver operating characteristic curve | 0-1 | Higher |

---

## 6. Experiment Matrix

The full experiment matrix covers **160 experiments** (8 models × 4 tasks × 5 steps).

### Priority Matrix (Current Sprint)

| | Translation | Summary | Bail | Verdict |
|---|---|---|---|---|
| **SEDD** | ☐ 0/F/FT/R/A | ☐ 0/F/FT/R/A | ☐ 0/F/FT/R/A | ☐ 0/F/FT/R/A |
| **LLaDA** | ☐ 0/F/FT/R/A | ☐ 0/F/FT/R/A | ☐ 0/F/FT/R/A | ☐ 0/F/FT/R/A |
| **LLaMA** | ☐ 0/F/FT/R/A | ☐ 0/F/FT/R/A | ☐ 0/F/FT/R/A | ☐ 0/F/FT/R/A |
| **Gemma** | ☐ 0/F/FT/R/A | ☐ 0/F/FT/R/A | ☐ 0/F/FT/R/A | ☐ 0/F/FT/R/A |

### Full Matrix (Phase 2)

| | Translation | Summary | Bail | Verdict |
|---|---|---|---|---|
| D3PM | ☐ | ☐ | ☐ | ☐ |
| DiffuseLM | ☐ | ☐ | ☐ | ☐ |
| Mistral | ☐ | ☐ | ☐ | ☐ |
| IndicBERT | ☐ | ☐ | ☐ | ☐ |

---

## 7. Technical Details

### 7.1 Diffusion vs AR: Key Architectural Differences

| Aspect | Diffusion Models | Auto-Regressive Models |
|--------|-----------------|----------------------|
| **Generation Direction** | All tokens simultaneously | Left-to-right, one at a time |
| **Attention** | Bidirectional (full) | Causal (unidirectional) |
| **Training Objective** | Denoising (predict clean from noisy) | Next-token prediction |
| **Generation Speed** | Multiple passes, but parallel per pass | Sequential, but single pass |
| **Controllability** | Natural (via guidance) | Requires constrained decoding |
| **Infilling** | Native support | Requires specialized training |
| **Token Dependencies** | Global (all tokens see all) | Only left context |

### 7.2 Server Configuration

- **Server:** 172.31.100.251 (IIT Kanpur)
- **Directory:** /data/debkanta/
- **Subdirectories:**
  - `/data/debkanta/Hindi_translation/`
  - `/data/debkanta/Hindi_summary/`
  - `/data/debkanta/Hindi_legal/`
  - `/data/debkanta/Hindi_judgeprediction/`

---

## 8. Results

### 8.1 Results Tables

_Results will be populated as experiments are completed._

### 8.2 Key Comparisons

1. **Diffusion vs AR overall:** Which paradigm performs better on average?
2. **Task-specific:** Does diffusion excel on certain tasks (e.g., infilling-heavy)?
3. **Pipeline progression:** How much does fine-tuning help each paradigm?
4. **Efficiency:** Speed/quality trade-offs between paradigms

---

## 9. References

1. Lou, A., Meng, C., & Ermon, S. (2024). "Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution." ICML 2024.
2. Nie, S. et al. (2025). "Large Language Diffusion Models." arXiv:2502.09992.
3. Austin, J. et al. (2021). "Structured Denoising Diffusion Models in Discrete State-Spaces." NeurIPS 2021.
4. Li, X. L. et al. (2022). "Diffusion-LM Improves Controllable Text Generation." NeurIPS 2022.
5. Touvron, H. et al. (2024). "LLaMA 3." Meta AI.
6. Gemma Team (2024). "Gemma: Open Models Based on Gemini." Google DeepMind.
7. Jiang, A. Q. et al. (2023). "Mistral 7B." Mistral AI.
8. Kakwani, D. et al. (2023). "IndicNLPSuite." AI4Bharat.
9. Kapoor, A. et al. (2022). "HLDC: Hindi Legal Documents Corpus." Exploration Lab.
10. Kunchukuttan, A. et al. (2021). "IIT Bombay English-Hindi Parallel Corpus." CFILT.
