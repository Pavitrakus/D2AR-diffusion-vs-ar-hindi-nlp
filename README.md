# D2AR: Diffusion versus Autoregressive Language Models for Hindi NLP

![Python 3.10](https://img.shields.io/badge/python-3.10-blue) ![PyTorch 2.1](https://img.shields.io/badge/pytorch-2.1-orange) ![License: Research](https://img.shields.io/badge/license-research--use-lightgrey)

**Principal Investigator:** Prof. Debkanta Chakraborty
**Research team:** Aditya Bhatia, Tanish Anand, Pavitra Kushwaha

D2AR is a systematic, reproducible benchmark that compares eight open language models from two generation paradigms, text diffusion and autoregressive decoding, on four Hindi NLP tasks across a five-stage adaptation protocol. The repository provides a unified evaluation harness that exposes heterogeneous architectures through a single model interface, and emits the full set of raw outputs, metrics, comparison reports, and publication-grade figures for the complete 8 x 4 x 5 experiment matrix.

## Abstract

Autoregressive language models generate text left-to-right by factorizing the joint token distribution with a causal objective. Text diffusion language models take a fundamentally different route: they corrupt a sequence with a structured noise process and learn to reverse it, generating all positions in parallel and refining them over multiple denoising steps. Diffusion models have reached competitive quality at scale (Lou et al., 2024; Nie et al., 2025), yet nearly all published evidence concerns English and other high-resource languages. Hindi, the third most spoken language in the world, remains severely under-resourced in NLP research, and the intersection of diffusion language modeling with Hindi is essentially unstudied. Legal Hindi is even less explored, with no prior application of diffusion language models to Hindi legal documents.

We address this gap with a controlled comparison of four diffusion language models, SEDD, LLaDA, D3PM, and Diffusion-LM, against four autoregressive baselines, LLaMA 3 8B Instruct, Gemma 2 9B Instruct, Mistral 7B Instruct, and IndicBERT v2. Models are evaluated on English-to-Hindi machine translation, Hindi abstractive summarization, and two legal classification tasks (bail prediction and judge verdict prediction) drawn from the HLDC corpus of over 900,000 Hindi legal documents. Each model is run through five adaptation stages, zero-shot prompting, few-shot prompting, parameter-efficient fine-tuning, retrieval-augmented generation, and agentic prompting, yielding a 160-cell experiment matrix. To our knowledge this is the first systematic benchmark of text diffusion models for Hindi and the first application of diffusion language models to Hindi legal document analysis.

## Research questions

> Can text diffusion models match or outperform autoregressive models on Hindi NLP tasks, and under what conditions does each paradigm excel?

The benchmark is designed to answer four specific questions:

1. **Paradigm-level gap.** Does either paradigm dominate on average across the task suite, or does the winner depend on task structure?
2. **Task structure.** Does the bidirectional, globally conditioned objective of diffusion models confer an advantage on tasks where full-context reasoning matters, such as legal classification over long documents, relative to left-to-right decoding?
3. **Adaptation dynamics.** How much does each adaptation stage (prompting, fine-tuning, retrieval, agentic scaffolding) close the gap between paradigms?
4. **Efficiency.** What is the latency and compute trade-off of multi-pass denoising against single-pass autoregressive decoding at comparable quality?

## Contributions

1. **First diffusion-vs-AR benchmark for Hindi.** A systematic 160-cell comparison of four diffusion and four autoregressive language models on four Hindi NLP tasks.
2. **First diffusion LMs on Hindi legal NLP.** Application of text diffusion models to bail prediction and judge verdict prediction over the HLDC corpus.
3. **Unified interface over heterogeneous architectures.** A common `BaseModel` interface with standardized `GenerationOutput` and `ClassificationOutput` contracts over six distinct decoding mechanisms (causal decoding, masked-language modeling, score-based discrete diffusion, masking diffusion, structured-transition diffusion, and continuous embedding-space diffusion).
4. **Five-stage adaptation protocol.** A reproducible evaluation ladder from zero-shot prompting through QLoRA fine-tuning, retrieval-augmented generation, and agentic prompting, quantifying how adaptation changes the paradigm gap.
5. **Reproducibility artifacts.** Raw predictions, metric JSON, aggregated reports, and comparison plots emitted automatically for every experiment, with resumable batch execution.

## Models under study

Models are chosen as the strongest publicly available checkpoints in each family. The diffusion set spans both discrete and continuous formulations; the autoregressive set spans the dominant open decoder-only checkpoints plus an Indic-specific encoder. Where a paradigm has no large Hindi-specific checkpoint, the two sides are compared at their respective strongest general-purpose checkpoints and the parameter-scale mismatch is treated as a stated limitation rather than ignored.

| Model | Paradigm | Parameters | Context | Mechanism |
|---|---|---|---|---|
| SEDD | discrete diffusion | ~350M | 1024 | Score-entropy ratio estimation, log-linear schedule, 64 denoising steps |
| LLaDA | masking diffusion | 8B | 4096 | Confidence-based progressive unmasking, 64 denoising steps |
| D3PM | discrete diffusion | ~110M | 256 | Structured absorbing-state transitions, cosine beta schedule, 100 steps |
| Diffusion-LM | continuous diffusion | ~100M | 128 | Gaussian noise in embedding space, 200 steps, nearest-token rounding |
| LLaMA 3 8B Instruct | autoregressive | 8B | 8192 | Causal decoding with grouped-query attention |
| Gemma 2 9B Instruct | autoregressive | 9B | 8192 | Causal decoding with alternating local and global attention |
| Mistral 7B Instruct v0.3 | autoregressive | 7B | 32768 | Causal decoding with sliding-window attention |
| IndicBERT v2 | masked LM | 278M | 512 | Masked-language pretraining over Indian languages (AI4Bharat) |

### Diffusion language models

**SEDD (Score Entropy Discrete Diffusion).** SEDD learns discrete diffusion by estimating the ratios of the data distribution rather than the conventional denoising probabilities, using a score-entropy training objective (Lou et al., 2024). Generation starts from a fully noised sequence and iteratively refines it by predicting discrete score functions at 64 steps under a log-linear noise schedule. SEDD was the first discrete diffusion model to outperform GPT-2 on language modeling perplexity. The wrapper supports both the official Hugging Face checkpoint and a simplified bidirectional-encoder fallback with an absorbing-state forward process.

**LLaDA (Large Language Diffusion with mAsking).** LLaDA trains an 8B-parameter Transformer under a masking-based diffusion objective: a forward process randomly masks tokens, and the reverse process predicts masked tokens in parallel, unmasking the highest-confidence positions at each step (Nie et al., 2025). LLaDA reports competitiveness with LLaMA 3 8B on general, math, and code benchmarks and is evaluated here under 4-bit NF4 quantization with Flash Attention 2.

**D3PM (Structured Denoising Diffusion Models in Discrete State-Spaces).** D3PM generalizes Gaussian diffusion to categorical token spaces by defining structured Markov transition matrices for the corruption process (Austin et al., 2021). This work is the bridge between diffusion and masked language models, since the absorbing-state transition recovers masked token prediction as a special case. The wrapper uses an absorbing transition with a cosine beta schedule over 100 denoising steps.

**Diffusion-LM.** Diffusion-LM operates in continuous embedding space: Gaussian noise is added to word embeddings, a Transformer denoiser predicts the noise, and the denoised vectors are rounded to the nearest vocabulary token by cosine similarity (Li et al., 2022). The original formulation also supports gradient-based controllable generation through plug-and-play classifiers.

### Autoregressive baselines

**LLaMA 3 8B Instruct** (Meta AI) is a causal decoder with grouped-query attention and an 8K context, selected as the flagship open autoregressive model. **Gemma 2 9B Instruct** (Google DeepMind) alternates local sliding-window and global attention every other layer and is trained with distillation from a larger model; it performs well on Indic languages. **Mistral 7B Instruct v0.3** (Mistral AI) uses sliding-window attention with a 32K context. **IndicBERT v2** (AI4Bharat) is a 278M-parameter masked language model pretrained on Indian languages through the IndicNLPSuite effort and serves as the Indic-specific encoder baseline. The three decoder-only models are served through the Transformers causal-LM path with NF4 4-bit quantization; IndicBERT is served as a masked LM with cloze-style scoring.

## Tasks and datasets

| # | Task | Task type | Dataset | Primary metric |
|---|---|---|---|---|
| 1 | English to Hindi translation | sequence-to-sequence | IIT Bombay English-Hindi Parallel Corpus | BLEU |
| 2 | Hindi abstractive summarization | text generation | XL-Sum (Hindi), ILSUM | ROUGE-1 |
| 3 | Bail prediction | binary classification | HLDC | macro F1 |
| 4 | Judge verdict prediction | multi-class classification | HLDC, IndianBailJudgments-1200 | macro F1 |

**English to Hindi translation.** The IIT Bombay English-Hindi parallel corpus provides roughly 1.49 million sentence pairs collected from news, Wikipedia, and government sources (Kunchukuttan et al., 2018). Evaluation uses the reference-level protocol with BLEU, chrF, BERTScore, COMET, and TER as the declared metric set. Samanantar (Ramesh et al., 2022) and OPUS-100 are available as backup sources.

**Hindi abstractive summarization.** The Hindi split of XL-Sum (Hasan et al., 2021), roughly 12,000 BBC Hindi news articles with reference summaries, is the primary source, with ILSUM and IndicSentenceSummarization as alternatives. The declared metric set is ROUGE-1, ROUGE-2, ROUGE-L, BERTScore, and BLEU.

**Bail prediction.** The Hindi Legal Documents Corpus (HLDC) contains over 900,000 legal documents from district courts in Uttar Pradesh, annotated for bail prediction (Kapoor et al., 2022). The task is binary classification between bail granted and bail rejected, scored by accuracy, macro and weighted F1, per-class precision and recall, and AUC-ROC.

**Judge verdict prediction.** Verdict prediction operates over HLDC supplemented by IndianBailJudgments-1200, a set of 1,200 annotated bail judgments with structured attributes (Deshmukh et al., 2023). The label space is three-way (convicted, acquitted, remanded) and the declared metric set is accuracy, macro and weighted F1, per-class precision and recall, and the confusion matrix.

Hindi adds challenges that an English-centric benchmark does not expose. The Devanagari abugida renders word boundaries and morphological segmentation ambiguous, so character-level and embedding-based metrics (chrF, BERTScore) are treated as complementary to surface n-gram overlap (BLEU, ROUGE). Legal documents compound this with Sanskrit-derived vocabulary, IPC section references, English code-mixing, and long input lengths.

## Evaluation framework

**Generation metrics.** BLEU is computed with sacrebleu using the international tokenization mode; chrF is computed at the character level; ROUGE-1/2/L are computed with the standard rouge-score F-measures; BERTScore uses `bert-base-multilingual-cased` with Hindi as the target language; COMET and TER are part of the declared protocol for translation. Perplexity of generated text is measured with a reference language model (IndicBARTSS) or, when that checkpoint is unavailable, a text-statistics estimator.

**Classification metrics.** Accuracy, macro and weighted F1, per-class precision and recall, and the confusion matrix are computed with scikit-learn over normalized labels, with a label-matching step that maps free-form model responses onto the closed class set.

All metric implementations return standardized `MetricResult` objects with score and detail fields, serialized per experiment into `metrics.json`.

## Adaptation pipeline

Each model is evaluated through five successive adaptation stages that isolate the contribution of prompting, optimization, retrieval, and reasoning scaffolding.

**Stage 1, zero-shot.** Direct inference with a task instruction and no Hindi examples. This stage isolates the model's inherent multilingual competence.

**Stage 2, few-shot.** Five carefully curated Hindi demonstrations are prepended to each prompt with no weight updates. This isolates in-context learning.

**Stage 3, parameter-efficient fine-tuning.** Decoder-only models (LLaMA, Gemma, Mistral) are fine-tuned with LoRA on top of NF4 4-bit quantization (QLoRA): rank 16, alpha 32, dropout 0.05, learning rate 2e-4, three epochs, gradient accumulation 4, linear warmup ratio 0.1, weight decay 0.01, fp16. Low-rank adapters target the query, key, value, output, and gating projections. IndicBERT is fine-tuned with a full sequence-classification head. Diffusion models use a custom denoising-loss loop: a random timestep is sampled per batch, the forward noising process is applied, and the model is trained to reconstruct the clean sequence by cross-entropy on denoised logits (or mean-squared error in embedding space for Diffusion-LM), with AdamW, cosine annealing, and gradient clipping at norm 1.0.

**Stage 4, retrieval-augmented generation.** Documents are chunked into 512-token windows with 50 tokens of overlap, embedded with `paraphrase-multilingual-MiniLM-L12-v2`, and indexed in a FAISS `IndexFlatIP` over L2-normalized vectors, giving cosine similarity retrieval with top-k 5. Retrieved context is prepended to the task prompt. A keyword-matching fallback keeps the pipeline functional without the embedding stack.

**Stage 5, agentic prompting.** Multi-step prompting with chain-of-thought reasoning, self-consistency, and answer refinement is applied on top of the adapted model.

## Experiment matrix

The full matrix is 8 models x 4 tasks x 5 stages, 160 cells. A priority matrix reduces the first pass to the four headline models, SEDD, LLaDA, LLaMA 3, and Gemma 2, across all tasks and stages (80 cells), so that results for the strongest representatives of each paradigm are available early. The batch runner skips completed cells on resume, isolates per-cell failures, and writes aggregated results, a cross-tabulated comparison report, and figures (model-by-task heatmaps, grouped bar charts, and per-task pipeline progression plots) into `results/`.

## Results

Raw outputs and metrics are written per experiment to `results/raw/<task>/<model>/<stage>/` as `raw_output.json` and `metrics.json`. Aggregated results, the comparison report, and figures are written to `results/metrics/aggregated/` and `results/figures/` as experiments complete. Tables are populated from these artifacts as the matrix progresses.

## Quick start

### Requirements

- Python 3.10+
- CUDA 11.8+ or 12.1+
- A GPU with 24 GB or more VRAM for the 8B-class models under 4-bit quantization (an A100/H100-class node is recommended for fine-tuning)
- A Hugging Face account with access accepted for gated checkpoints (LLaMA 3, Gemma 2)

### Installation

```bash
git clone https://github.com/Pavitrakus/D2AR-diffusion-vs-ar-hindi-nlp.git
cd D2AR-diffusion-vs-ar-hindi-nlp

python -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

Copy `.env.example` to `.env` and set `HF_TOKEN` for the gated models and datasets:

```bash
cp .env.example .env
# HF_TOKEN=hf_...
```

### Running experiments

Run a single experiment (one model, one task, one stage):

```bash
python experiments/run_experiment.py \
    --model llama \
    --task hindi_translation \
    --pipeline zero_shot
```

Run the priority matrix (SEDD, LLaDA, LLaMA, Gemma across all tasks and stages):

```bash
python experiments/run_all_experiments.py --priority-only
```

Run the full 160-cell matrix:

```bash
python experiments/run_all_experiments.py --full
```

Target a custom subset:

```bash
python experiments/run_all_experiments.py \
    --models sedd mistral \
    --tasks bail translation \
    --steps zero_shot few_shot
```

Additional flags include `--num_samples`, `--max_new_tokens`, `--temperature`, `--device`, `--no-4bit` (disable quantization), and `--no-resume` (re-run completed cells).

## Project structure

```
config/                  # experiment taxonomy and per-model, per-task configuration
  base_config.py         #   ModelName, TaskName, PipelineStep enums; ExperimentConfig
  model_configs.py       #   Hugging Face IDs, LoRA targets, generation defaults per model
  task_configs.py        #   datasets, prompts, few-shot exemplars, metric sets per task
src/
  models/
    base_model.py        # unified BaseModel interface, GenerationOutput / ClassificationOutput
    diffusion/           # SEDD, LLaDA, D3PM, Diffusion-LM wrappers
    autoregressive/      # LLaMA, Gemma, Mistral, IndicBERT wrappers
  pipelines/
    fine_tuning.py       # LoRA / QLoRA for decoders, custom denoising loop for diffusion
  evaluation/
    metrics.py           # BLEU, chrF, ROUGE, BERTScore, perplexity, classification metrics
  rag/
    rag_chain.py         # multilingual embeddings, FAISS cosine index, prompt augmentation
  data/
    data_loader.py       # Hugging Face dataset loading with fallback sample data
  utils/
    visualization.py     # heatmaps, bar charts, radar charts, confusion matrices
experiments/
  run_experiment.py      # single-cell runner (model x task x stage)
  run_all_experiments.py # matrix runner, resume, aggregation, figures
docs/
  research_document.md   # full research document
  dataset_descriptions.md
  Execution_Guide_For_Debkanta.txt
requirements.txt
.env.example
```

The `results/` tree is created and populated at runtime under `results/raw/`, `results/metrics/`, and `results/figures/`.

## References

1. Lou, A., Meng, C., and Ermon, S. Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution. *ICML*, 2024. arXiv:2310.16834.
2. Nie, S., Zhu, F., You, Z., Zhang, X., Ou, J., Hu, J., Zhou, J., Lin, Y., Wen, J.-R., and Li, C. Large Language Diffusion Models. 2025. arXiv:2502.09992.
3. Austin, J., Johnson, D., Ho, J., Simonyan, K., and van den Berg, E. Structured Denoising Diffusion Models in Discrete State-Spaces. *NeurIPS*, 2021. arXiv:2107.03006.
4. Li, X. L., Thickstun, J., Gulrajani, I., Liang, P., and Hashimoto, T. B. Diffusion-LM Improves Controllable Text Generation. *NeurIPS*, 2022. arXiv:2205.14217.
5. Touvron, H., et al. The Llama 3 Herd of Models. 2024. arXiv:2407.21783.
6. Gemma Team. Gemma 2: Improving Open Language Models at a Practical Size. 2024. arXiv:2408.00118.
7. Jiang, A. Q., et al. Mistral 7B. 2023. arXiv:2310.06825.
8. Kakwani, D., et al. IndicNLPSuite: Monolingual Corpora, Evaluation Benchmarks and Pre-trained Models for Indian Languages. *Findings of EMNLP*, 2020.
9. Kapoor, A., et al. HLDC: Hindi Legal Documents Corpus. Exploration Lab, IIT Kanpur, 2022.
10. Deshmukh, S., et al. IndianBailJudgments: A Dataset of Annotated Indian Bail Judgments. 2023.
11. Kunchukuttan, A., Mehta, P., and Bhattacharyya, P. The IIT Bombay English-Hindi Parallel Corpus. *LREC*, 2018.
12. Hasan, T., et al. XL-Sum: Large-Scale Multilingual Abstractive Summarization for 44 Languages. *ACL*, 2021.
13. Ramesh, G., et al. Samanantar: The Largest Publicly Available Parallel Corpora Collection for 11 Indic Languages. *TACL*, 2022.

## Citation

If you use this repository in your research, please cite it as:

```bibtex
@misc{chakraborty2026d2ar,
  title = {D2AR: Diffusion versus Autoregressive Language Models for Hindi NLP},
  author = {Chakraborty, Debkanta and Bhatia, Aditya and Anand, Tanish and Kushwaha, Pavitra},
  year = {2026},
  note = {Technical report}
}
```

## License and acknowledgements

This repository is released for academic research purposes under the supervision of Prof. Debkanta Chakraborty. The base models and datasets retain their respective licenses; please verify access terms for the gated checkpoints (LLaMA 3, Gemma 2) and the HLDC corpus before use.

The benchmark builds on the work of the authors of SEDD (Stanford), LLaDA (Renmin University of China), D3PM (Google Research), Diffusion-LM (Stanford), LLaMA (Meta AI), Gemma (Google DeepMind), Mistral (Mistral AI), and IndicBERT (AI4Bharat), and on the IIT Bombay parallel corpus (CFILT), the Hindi Legal Documents Corpus (Exploration Lab), and XL-Sum (BBC Hindi). Compute was provided by the research infrastructure at IIT Kanpur.
