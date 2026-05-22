# Dataset Documentation

## 1. IIT Bombay English-Hindi Parallel Corpus

- **Source:** CFILT Lab, IIT Bombay
- **HuggingFace:** `cfilt/iitb-english-hindi`
- **Size:** ~1.49 million sentence pairs
- **Description:** Parallel corpus of English-Hindi sentence pairs collected from various sources including news, Wikipedia, and government documents.
- **Usage:** English→Hindi translation task
- **License:** CC-BY-4.0

## 2. XL-Sum Hindi

- **Source:** BBC Hindi, crawled by researchers
- **HuggingFace:** `csebuetnlp/xlsum` (config: `hindi`)
- **Size:** ~12,000 articles with summaries
- **Description:** News articles in Hindi with professionally written summaries
- **Usage:** Hindi text summarization task
- **License:** CC-BY-NC-SA-4.0

## 3. HLDC (Hindi Legal Documents Corpus)

- **Source:** Exploration Lab, IIT Kanpur
- **HuggingFace:** `Exploration-Lab/HLDC`
- **Size:** 900,000+ legal documents
- **Description:** Hindi legal documents from district courts in Uttar Pradesh, annotated for bail prediction
- **Usage:** Bail prediction + judge verdict prediction
- **License:** Research use only
- **Paper:** Kapoor et al. (2022)

## 4. IndianBailJudgments-1200

- **Source:** Sneha Deshmukh et al.
- **HuggingFace:** `SnehaDeshmukh/IndianBailJudgments-1200`
- **Size:** 1,200 annotated bail judgments
- **Description:** 20+ structured attributes per case including bail outcome, IPC sections, crime type
- **Usage:** Supplementary data for verdict prediction
- **License:** Research use only

## 5. Alternative Datasets

| Dataset | Usage | HuggingFace ID |
|---------|-------|----------------|
| Samanantar | Translation (backup) | `ai4bharat/samanantar` |
| OPUS-100 | Translation (backup) | `Helsinki-NLP/opus-100` |
| IL-TUR | Legal NLP | `Exploration-Lab/IL-TUR` |
| Varta | Summarization (backup) | `rahular/varta` |

## Data Access Instructions

```bash
# Install datasets library
pip install datasets

# Download a specific dataset
python -c "from datasets import load_dataset; ds = load_dataset('cfilt/iitb-english-hindi', split='test[:100]'); print(ds)"
```

For the HLDC dataset, you may need to accept the license on HuggingFace before downloading.
