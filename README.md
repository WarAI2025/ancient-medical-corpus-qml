# Ancient Medical Corpus — Febrile Crisis Semiotics
### Hybrid Classical-Quantum NLP across Five Ancient Medical Traditions

[![License: CC-BY 4.0](https://img.shields.io/badge/Data-CC--BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![License: MIT](https://img.shields.io/badge/Code-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-green.svg)](https://www.python.org/)
[![PennyLane](https://img.shields.io/badge/QML-PennyLane-black.svg)](https://pennylane.ai/)

---

## Overview

This repository contains the data and code associated with:

> [Author] (2026). *Hybrid classical-quantum NLP applied to ancient medical texts: a multicultural corpus of 160 cases for the computational study of crisis semiotics in febrile prognosis.* Submitted to PLOS ONE.

**Two statistically significant findings:**
- **H4 — Clinical prognosis from sweat description:** 71.2% LOO-CV accuracy across 5 traditions, p = 3.79×10⁻⁸
- **H6 — Extremity temperature classification:** 58.8% LOO-CV accuracy across 5 traditions, p = 1.62×10⁻²

---

## Repository structure

```
ancient-medical-corpus/
│
├── README.md                        ← this file
│
├── data/
│   ├── corpus_ancient_medical.json  ← full annotated corpus (160 cases, metadata)
│   └── corpus_ancient_medical.csv   ← flat CSV for tabular analysis
│
├── code/
│   └── QML_Complete_Standalone.py   ← complete self-contained pipeline
│                                       (auto-detects completed experiments,
│                                        runs only missing ones)
│
├── docs/
│   ├── annotation_guidelines.md     ← decision rules for all 8 variables
│   └── results_summary.md           ← master results table (all 15 conditions)
│
└── figures/
    └── [figures available on request / from Drive]
```

---

## Corpus overview

| Tradition  | Period        | Primary source                                    | N  |
|------------|---------------|---------------------------------------------------|----|
| Greek      | ~5th c. BCE   | Hippocrates, Epidemics I–V; Prognostic; Aphorisms | 42 |
| Babylonian | ~10th c. BCE  | Diagnostic Handbook SA.GIG (Geller 2010)          | 28 |
| Galenic    | ~2nd c. CE    | Galen, De Crisibus (Johnston 2011)                | 30 |
| Islamic    | ~11th c. CE   | Avicenna, Canon of Medicine IV (Gruner 1930)      | 30 |
| Chinese    | ~3rd c. CE    | Shang Han Lun; Jingui Yaolue (Luo 1993)           | 30 |
| **Total**  |               |                                                   | **160** |

---

## Annotation variables

### Layer 1 — Primary annotation (H1–H5)

| Code | Variable              | Classes                                      | Coverage |
|------|-----------------------|----------------------------------------------|----------|
| H1   | Anatomical location   | head / thorax / extremities / general        | 100%     |
| H2   | Sweat temperature     | cold / tepid / warm                          | 100%     |
| H3   | Critical-day timing   | critical day / non-critical                  | 100%     |
| H4   | Clinical prognosis ★  | favorable / fatal                            | 100%     |
| H5   | Tradition of origin   | Greek / Babylonian / Galenic / Islamic / Chinese | 100% |

### Layer 2 — Extended annotation (H6–H8)

| Code | Variable              | Classes                                      | Coverage |
|------|-----------------------|----------------------------------------------|----------|
| H6   | Extremity temperature ★ | warm / cold                                | 100%     |
| H7   | Delirium during crisis | absent / present                            | 100%     |
| H8   | Fever pattern         | continuous / intermittent / remittent        | 100%     |

★ = statistically significant in LOO-CV experiments

---

## Master results — all 15 experimental conditions

| Hyp. | Corpus        | Ablation | N   | QML acc. | SVM acc. | RF acc. | p-value    | Sig. |
|------|---------------|----------|-----|----------|----------|---------|------------|------|
| H4   | All traditions | No      | 160 | 71.2%    | 91.9%    | 88.8%   | 3.79×10⁻⁸  | ★    |
| H4   | Greek only     | No      | 42  | 90.5%    | 95.2%    | 92.9%   | <0.0001    | ★    |
| H4   | Greek only     | Yes     | 42  | 76.2%    | 81.0%    | 73.8%   | 4.70×10⁻⁴  | ★    |
| H6   | All traditions | No      | 160 | 58.8%    | 75.6%    | 75.6%   | 1.62×10⁻²  | ★    |
| H4   | All traditions | Yes     | 160 | 52.5%    | 85.6%    | 82.5%   | 2.90×10⁻¹  | —    |
| H6   | Greek only     | No      | 42  | 57.1%    | 75.6%    | 75.6%   | 2.20×10⁻¹  | —    |
| H5   | All traditions | No      | 160 | 23.8%    | 68.8%    | 66.2%   | 1.39×10⁻¹  | —    |
| H2   | All traditions | No      | 160 | 33.8%    | 78.1%    | 75.6%   | 4.85×10⁻¹  | —    |
| H2   | Greek only     | No      | 42  | 19.1%    | 66.7%    | 64.3%   | 9.87×10⁻¹  | —    |
| H1   | All traditions | No      | 160 | 20.6%    | 65.6%    | 66.9%   | 9.17×10⁻¹  | —    |
| H1   | Greek only     | No      | 42  | 23.8%    | 61.9%    | 54.8%   | 6.29×10⁻¹  | —    |
| H3   | All traditions | No      | 160 | 27.5%    | 94.4%    | 94.4%   | 1.00×10⁰   | —    |
| H3   | Greek only     | No      | 42  | 52.4%    | 88.1%    | 85.7%   | 4.39×10⁻¹  | —    |
| H7   | All traditions | No      | 160 | 41.2%    | 93.8%    | 93.8%   | 9.89×10⁻¹  | —    |
| H8   | All traditions | No      | 160 | 20.6%    | 88.1%    | 88.1%   | 1.00×10⁰   | —    |

---

## Quick start

```python
import json

# Load corpus
with open('data/corpus_ancient_medical.json') as f:
    corpus = json.load(f)

cases = corpus['cases']
print(f"Total cases: {len(cases)}")

# Filter by tradition
greek = [c for c in cases if c['tradicion'] == 'griega']
print(f"Greek cases: {len(greek)}")

# Filter favorable prognosis
favorable = [c for c in cases if c['h4_pronostico'] == 0]
print(f"Favorable outcomes: {len(favorable)}")

# Filter cold extremities
cold_ext = [c for c in cases if c.get('h11_extremid') == 1]
print(f"Cold extremities: {len(cold_ext)}")
```

---

## Running the full pipeline

The pipeline is designed for **Google Colab with a T4 GPU**. It auto-detects completed experiments from Drive and trains only missing ones — safe to re-run at any time.

```bash
# 1. Open Google Colab
# 2. Create a new notebook
# 3. Paste the entire contents of code/QML_Complete_Standalone.py into a cell
# 4. Run — the script will:
#    - Install all dependencies automatically
#    - Mount Google Drive
#    - Scan for completed experiments
#    - Train only missing conditions
#    - Save results, figures, and JSON to Drive
```

**Estimated runtime:** ~1.5h per experiment (160 LOO folds × ~28s/fold) on T4 GPU.  
**Total for all 12 conditions:** ~14h (with checkpointing, resumable if interrupted).

---

## Pipeline architecture

```
Input text (English translation)
        ↓
Bio_ClinicalBERT [CLS] embedding (768-dim, L2-normalized)
        ↓
Classical compression: Linear(768→32) → BN → GELU → Dropout → Linear(32→4) → Tanh
        ↓
4-qubit VQC: AngleEmbedding + StronglyEntanglingLayers (12 params)
        ↓
Decoder: Linear(4→8) → ReLU → Linear(8→N_classes)
        ↓
LOO-CV prediction
```

**Baselines:** SVM-RBF and Random Forest, both under LOO-CV (parallel via joblib).  
**Statistics:** Binomial p-value vs. 1/N_classes random baseline, Wilson 95% CI, balanced accuracy.

---

## Data format

Each case in `corpus_ancient_medical.json` follows this structure:

```json
{
  "id": "Gr_Ep1_C1",
  "fuente": "Hippocrates, Epidemics I",
  "tradicion": "griega",
  "siglo": -5,
  "texto": "Philiscus lived by the wall...",
  "h1_localizacion": 2,
  "h2_temperatura": 0,
  "h3_momento": 0,
  "h4_pronostico": 1,
  "h5_tradicion": 0,
  "h9_fiebre": 0,
  "h10_delirio": 0,
  "h11_extremid": 1,
  "h12_resp": 1,
  "h13_apetito": 1
}
```

---

## Known limitations and annotation notes

- All annotations based on **English translations** of primary sources
- **Single annotator** (author); annotation guidelines published for independent re-annotation
- One case flagged as philological ambiguity: **Gr_Prog_C1** (Hippocratic Prognostic) — functions as normative rule rather than individual case report; misclassified in all experimental conditions
- H6 (extremity temperature): ~64% of cases have explicit lexical basis; ~36% annotated by implication
- H7 (delirium): ~6% positive rate — severe class imbalance
- H8 (fever pattern): ~92% continuous — corpus construction artifact

---

## Citation

```bibtex
@article{[author]2026ancient,
  title   = {Hybrid classical-quantum NLP applied to ancient medical texts:
             a multicultural corpus of 160 cases for the computational study
             of crisis semiotics in febrile prognosis},
  author  = {[Author]},
  journal = {PLOS ONE},
  year    = {2026},
  note    = {Submitted. Preprint available at GitHub: [URL]}
}
```

---

## License

- **Data** (`data/`): [Creative Commons Attribution 4.0 International (CC-BY 4.0)](https://creativecommons.org/licenses/by/4.0/)
- **Code** (`code/`): [MIT License](https://opensource.org/licenses/MIT)

---

## Contact

[Author name]  
Independent researcher  
[email] · [ORCID]
