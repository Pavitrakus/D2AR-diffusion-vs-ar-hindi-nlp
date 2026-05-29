# 🔬 Diffusion vs Auto-Regressive Language  Models for Hindi NLP

> **A comprehensive PhD  research project comparing Text Diffusion Mo dels against Auto-Regressive Models across Hi ndi NLP tasks**

**Principal Investigator:**  Prof. Debkanta Chakraborty  
**Research Team: ** Aditya Bhatia, Tanish Anand, Pavitra Kushw aha

---

## 📖 Project Overview

This proj ect systematically compares **8 state-of-the- art language models** (4 diffusion-based + 4  auto-regressive) on **4 Hindi NLP tasks** thr ough a **5-step evaluation pipeline**, produc ing the most comprehensive benchmark of diffu sion language models for Hindi to date.

###  Models Under Study

| Category | Model | Para meters | Key Innovation |
|----------|------- |-----------|----------------|
| 🌀 Diffusi on | **SEDD** | ~350M | Score Entropy Discret e Diffusion |
| 🌀 Diffusion | **LLaDA** |  8B | Large Language Diffusion with Masking |
 | 🌀 Diffusion | **D3PM** | ~110M | Discret e Denoising Diffusion Probabilistic Models |
 | 🌀 Diffusion | **DiffuseLM** | ~100M | Co ntinuous Diffusion with Gradient Control |
|  ➡️ Auto-Regressive | **LLaMA 3** | 8B | M eta's flagship open-source LLM |
| ➡️ Aut o-Regressive | **Gemma 2** | 9B | Google Deep Mind's efficient LLM |
| ➡️ Auto-Regressi ve | **Mistral** | 7B | Sliding window attent ion |
| ➡️ Auto-Regressive | **IndicBERT* * | 278M | AI4Bharat's Indic language model | 

### Tasks

| # | Task | Type | Dataset |
|- --|------|------|---------|
| 1 | Hindi Trans lation (En→Hi) | Seq2Seq Generation | IIT B ombay Parallel Corpus |
| 2 | Hindi Summariza tion | Abstractive Generation | ILSUM / Hindi  XSUM |
| 3 | Bail Prediction | Binary Classi fication | HLDC (900K+ docs) |
| 4 | Judge Ve rdict Prediction | Multi-class Classification  | HLDC + IndianBailJudgments |

### Pipeline  Steps

```
Zero-Shot → Few-Shot → Fine-T uning (LoRA) → RAG → Agentic AI
```

---
 
## 🚀 Quick Start

### Prerequisites
- Pyt hon 3.10+
- CUDA 11.8+ (GPU required for mode l inference)
- 24GB+ VRAM recommended (for 8B  models with quantization)

### Installation
 
```bash
# Clone the repository
git clone <re po-url>
cd diffusion-vs-ar-hindi-nlp

# Creat e virtual environment
python -m venv venv
sou rce venv/bin/activate  # Linux/Mac
# venv\Scr ipts\activate  # Windows

# Install dependenc ies
pip install -r requirements.txt
```

###  Running Experiments

```bash
# Run a single e xperiment
python experiments/run_experiment.p y \
    --model llama \
    --task hindi_tran slation \
    --pipeline zero_shot

# Run all  experiments for priority models
python exper iments/run_all_experiments.py --priority-only 

# Run full 8×4×5 experiment matrix
python  experiments/run_all_experiments.py --full
`` `

---

## 📁 Project Structure

```
├─ ─ config/             # Configuration syste m
├── src/
│   ├── models/          # Model wrappers (diffusion/ + autoregres sive/)
│   ├── tasks/          # Task  implementations
│   ├── pipelines/       # 5-step pipeline implementations
│   � ��── data/           # Data loading & pre processing
│   ├── evaluation/     #  Metrics & evaluation
│   ├── rag/             # RAG components
│   └── util s/          # Utilities (logging, GPU, viz)
� ��── experiments/        # Experiment run ners
├── results/            # All outp uts (raw/, metrics/, figures/)
├── note books/          # Jupyter analysis notebooks
 ├── docs/               # Documentation 
└── tests/              # Unit tests
` ``

---

## 📊 Evaluation Metrics

### Gene ration Tasks (Translation & Summarization)
-  **BLEU** — N-gram overlap
- **ROUGE-1/2/L**  — Recall-oriented evaluation
- **BERTScore ** — Semantic similarity
- **Perplexity** � �� Fluency measure
- **chrF** — Character-l evel F-score

### Classification Tasks (Bail  & Verdict)
- **Accuracy** — Overall correct ness
- **F1 (Macro/Weighted)** — Balanced p erformance
- **Precision / Recall** — Per-c lass performance
- **AUC-ROC** — Ranking qu ality

---

## 📄 License

This project is  for academic research purposes under Prof. De bkanta Chakraborty's supervision.

## 🙏 Ac knowledgements

- Stanford University (SEDD)
 - Renmin University of China (LLaDA)
- Google  Research (D3PM)
- AI4Bharat (IndicBERT, Indi c NLP resources)
- IIT Bombay (Hindi parallel  corpus)
- Exploration Lab (HLDC dataset)
  
