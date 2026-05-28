# 🔬 Diffusion vs Auto-Regressive Language Models for Hindi NLP

> **A comprehensive PhD research project comparing Text Diffusion Models against Auto-Regressive Models across Hindi NLP tasks**

**Principal Investigator:** Prof. Debkanta Chakraborty  
**Research Team:** Aditya Bhatia, Tanish Anand, Pavitra Kushwaha

---

## 📖 Project Overview

This project systematically compares **8 state-of-the-art language models** (4 diffusion-based + 4 auto-regressive) on **4 Hindi NLP tasks** through a **5-step evaluation pipeline**, producing the most comprehensive benchmark of diffusion language models for Hindi to date.

### Models Under Study

| Category | Model | Parameters | Key Innovation |
|----------|-------|-----------|----------------|
| 🌀 Diffusion | **SEDD** | ~350M | Score Entropy Discrete Diffusion |
| 🌀 Diffusion | **LLaDA** | 8B | Large Language Diffusion with Masking |
| 🌀 Diffusion | **D3PM** | ~110M | Discrete Denoising Diffusion Probabilistic Models |
| 🌀 Diffusion | **DiffuseLM** | ~100M | Continuous Diffusion with Gradient Control |
| ➡️ Auto-Regressive | **LLaMA 3** | 8B | Meta's flagship open-source LLM |
| ➡️ Auto-Regressive | **Gemma 2** | 9B | Google DeepMind's efficient LLM |
| ➡️ Auto-Regressive | **Mistral** | 7B | Sliding window attention |
| ➡️ Auto-Regressive | **IndicBERT** | 278M | AI4Bharat's Indic language model |

### Tasks

| # | Task | Type | Dataset |
|---|------|------|---------|
| 1 | Hindi Translation (En→Hi) | Seq2Seq Generation | IIT Bombay Parallel Corpus |
| 2 | Hindi Summarization | Abstractive Generation | ILSUM / Hindi XSUM |
| 3 | Bail Prediction | Binary Classification | HLDC (900K+ docs) |
| 4 | Judge Verdict Prediction | Multi-class Classification | HLDC + IndianBailJudgments |

### Pipeline Steps

```
Zero-Shot → Few-Shot → Fine-Tuning (LoRA) → RAG → Agentic AI
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- CUDA 11.8+ (GPU required for model inference)
- 24GB+ VRAM recommended (for 8B models with quantization)

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd diffusion-vs-ar-hindi-nlp

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Running Experiments

```bash
# Run a single experiment
python experiments/run_experiment.py \
    --model llama \
    --task hindi_translation \
    --pipeline zero_shot

# Run all experiments for priority models
python experiments/run_all_experiments.py --priority-only

# Run full 8×4×5 experiment matrix
python experiments/run_all_experiments.py --full
```

---

## 📁 Project Structure

```
├── config/             # Configuration system
├── src/
│   ├── models/         # Model wrappers (diffusion/ + autoregressive/)
│   ├── tasks/          # Task implementations
│   ├── pipelines/      # 5-step pipeline implementations
│   ├── data/           # Data loading & preprocessing
│   ├── evaluation/     # Metrics & evaluation
│   ├── rag/            # RAG components
│   └── utils/          # Utilities (logging, GPU, viz)
├── experiments/        # Experiment runners
├── results/            # All outputs (raw/, metrics/, figures/)
├── notebooks/          # Jupyter analysis notebooks
├── docs/               # Documentation
└── tests/              # Unit tests
```

---

## 📊 Evaluation Metrics

### Generation Tasks (Translation & Summarization)
- **BLEU** — N-gram overlap
- **ROUGE-1/2/L** — Recall-oriented evaluation
- **BERTScore** — Semantic similarity
- **Perplexity** — Fluency measure
- **chrF** — Character-level F-score

### Classification Tasks (Bail & Verdict)
- **Accuracy** — Overall correctness
- **F1 (Macro/Weighted)** — Balanced performance
- **Precision / Recall** — Per-class performance
- **AUC-ROC** — Ranking quality

---

## 📄 License

This project is for academic research purposes under Prof. Debkanta Chakraborty's supervision.

## 🙏 Acknowledgements

- Stanford University (SEDD)
- Renmin University of China (LLaDA)
- Google Research (D3PM)
- AI4Bharat (IndicBERT, Indic NLP resources)
- IIT Bombay (Hindi parallel corpus)
- Exploration Lab (HLDC dataset)
